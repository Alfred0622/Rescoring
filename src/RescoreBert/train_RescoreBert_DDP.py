import os
import argparse
import json
import sys
sys.path.append('..')
import torch
from tqdm import tqdm
import logging
import torch
import random

from pathlib import Path
from torch.utils.data import DataLoader
from utils.Datasets import getRescoreDataset
from src_utils.LoadConfig import load_config
from utils.CollateFunc import RescoreBertBatch, MWERSampler, BatchSampler, MWERBatch
from utils.PrepareModel import prepare_RescoreBert
from torch.optim import AdamW
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parse_args = parser.parse_args()

mode = sys.argv[1]

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)



if (torch.cuda.is_available()):
    device = torch.device('cuda' , parse_args.local_rank)
else:
    device = torch.device('cpu')

if (len(sys.argv) != 2):
    assert(len(sys.argv) == 2), "python ./train_RescoreBert.py {MD,MWER,MWED}"

# distributed system


print(f'mode:{mode}')
use_MWER = False
use_MWED = False
if (mode == "MWER"):
    use_MWER = True
elif (mode == 'MWED'):
    use_MWED = True
else:
    mode = 'MD'

config = f'./config/RescoreBert.yaml'
args, train_args, recog_args = load_config(config)

setting = 'withLM' if (args['withLM']) else 'noLM'

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/RescoreBert/{args['dataset']}/{mode}/{setting}/train.log",
    filemode="w",
    format=FORMAT,
)


if (args['dataset'] in ['aishell2']):
    dev_set = 'dev_ios'
else:
    dev_set = 'dev'

with open(f"./data/{args['dataset']}/{setting}/50best/MLM/train/last_rescore_data.json") as f, \
     open(f"./data/{args['dataset']}/{setting}/50best/MLM/{dev_set}/rescore_data.json") as dev:
    train_json = json.load(f)
    valid_json = json.load(dev)

model, tokenizer = prepare_RescoreBert(args['dataset'], device)

train_dataset = getRescoreDataset(train_json, args['dataset'], tokenizer, topk = args['nbest'], mode = mode)
valid_dataset = getRescoreDataset(valid_json, args['dataset'], tokenizer, topk = args['nbest'], mode = mode)

# Multiprocessing
dist.init_process_group(backend='nccl')
dist.barrier()
world_size = dist.get_world_size()

model = model.to(device)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [parse_args.local_rank], output_device=parse_args.local_rank)

if (not os.path.exists(f"./log/RescoreBert/{args['dataset']}/{mode}/{setting}")):
    os.makedirs(f"./log/RescoreBert/{args['dataset']}/{mode}/{setting}")

optimizer = AdamW(model.parameters(), lr = float(train_args['lr']))

if (use_MWED or use_MWER):
    train_sampler = MWERSampler(train_dataset)
    valid_sampler = MWERSampler(valid_dataset)
    
    train_batch_sampler = BatchSampler(train_sampler, train_args['train_batch'])
    valid_batch_sampler = BatchSampler(valid_sampler, train_args['train_batch'])

    train_loader = DataLoader(
        dataset = train_dataset,
        # batch_size=train_args['train_batch'],
        batch_sampler = train_batch_sampler,
        collate_fn = MWERBatch,
        num_workers = 4
    )

    valid_loader = DataLoader(
        dataset = valid_dataset,
        # batch_size = train_args['train_batch'],
        batch_sampler = valid_batch_sampler, 
        collate_fn = MWERBatch,
        num_workers = 4
    )
else:
    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset)

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size=train_args['train_batch'],
        sampler = train_sampler,
        collate_fn=RescoreBertBatch,
        num_workers = 4
    )

    valid_loader = DataLoader(
        dataset = valid_dataset,
        batch_size=train_args['valid_batch'],
        sampler = valid_sampler,
        collate_fn=RescoreBertBatch,
        num_workers = 4
    )

weight = 1e-4
weight = torch.tensor(weight, dtype = torch.float32).to(device)

optimizer.zero_grad()

checkpoint_path = Path(f"./checkpoint/{args['dataset']}/RescoreBert/{setting}/{mode}/{args['nbest']}best")
checkpoint_path.mkdir(parents=True, exist_ok=True)


if (parse_args.local_rank == 0):
    torch.save(model.state_dict(), f"{checkpoint_path}/checkpoint_train_0.pt")

dist.barrier()
map_location = {"cuda:%d" % 0 : "cuda:%d" % parse_args.local_rank}
model.load_state_dict(torch.load(f"{checkpoint_path}/checkpoint_train_0.pt", map_location = map_location))


for e in range(train_args['epoch']):
    train_sampler.set_epoch(e)
    valid_sampler.set_epoch(e)
    model.train()

    min_val_loss = 1e8
    min_val_cer = 1e6

    logging_loss = torch.tensor([0.0], device = device)

    for i, data in enumerate(tqdm(train_loader, ncols = 100)):

        data['input_ids'] = data['input_ids'].to(device, non_blocking = True)
        data['attention_mask'] = data['attention_mask'].to(device, non_blocking = True)
        data['labels'] = data['labels'].to(device, non_blocking = True)
        data['wer'] = data['wer'].to(device, non_blocking = True)
        # data = {k : v.to(device) for k ,v in data.items()}

        # print(f"rank {parse_args.local_rank} at epoch {e + 1}:{data['input_ids'].shape}")

        output = model(
            input_ids = data['input_ids'],
            attention_mask = data['attention_mask'],
            labels = data['labels']
        )
        loss = output["loss"] / float(train_args['accumgrad'])        

        # MWER
        if (mode == 'MWER'):
            first_score = data['score'].to(device)

            combined_score = first_score + output['score']

            avg_error = data['avg_error'].to(device)

            # softmax seperately
            index = 0
            for nbest in data['nbest']:

                combined_score[index: index + nbest] = torch.softmax(
                    combined_score[index:index + nbest], dim = -1
                )
                
                index = index + nbest

            loss_MWER = first_score * (data['wer'] - avg_error)
            loss_MWER = torch.sum(loss_MWER) / torch.tensor(train_args['accumgrad'], dtype = torch.float32)

            loss = loss_MWER + 1e-4 * loss
        
        elif (mode == 'MWED'):
            with torch.autograd.set_detect_anomaly(True):
                first_score = data['score'].to(device)
                wer = data['wer'].clone()

                assert(first_score.shape == output['score'].shape), f"first_score:{first_score.shape}, score:{output['score'].shape}"

                combined_score = first_score + output['score'].clone()
                
                index = 0
                scoreSum = torch.tensor([]).to(device)
                werSum = torch.tensor([]).to(device)

                for nbest in data['nbest']:

                    score_sum = torch.sum(
                        combined_score[index: index + nbest].clone()
                    ).repeat(nbest)

                    wer_sum = torch.sum(
                        wer[index:index + nbest].clone()
                    ).repeat(nbest)

                    scoreSum = torch.cat([scoreSum, score_sum])
                    werSum = torch.cat([werSum, wer_sum])

                    index = index + nbest

                index = 0

                assert(scoreSum.shape == combined_score.shape), f"scoreSum:{scoreSum.shape} != combined_score:{combined_score}"
                assert(werSum.shape == combined_score.shape), f"werSum:{werSum.shape} != combined_score:{combined_score}"
                
                T = scoreSum / werSum # hyperparameter T
                # print(f'combined_score:{combined_score}')
                combined_score = combined_score / T
                # print(f'combined_score after scale:{combined_score}')

                for nbest in data['nbest']:
                    combined_score[index: index + nbest] = torch.softmax(
                        combined_score[index:index + nbest].clone(), dim = -1
                    )
                    wer[index: index + nbest] = torch.softmax(
                        wer[index:index + nbest].clone(), dim = -1
                    )
                
                    index = index + nbest
                
                # print(f'combined_score after scale & softmax:{combined_score}')

                loss_MWED =  wer * torch.log(combined_score)
                loss_MWED = torch.neg(torch.sum(loss_MWED)) / torch.tensor(train_args['accumgrad'], dtype = torch.float32)
                loss = loss_MWED + 1e-4 * loss

        loss.backward()

        if (((i + 1) % int(train_args['accumgrad'])) == 0):
            optimizer.step()
            optimizer.zero_grad()
        
        if ((i + 1) % int(train_args['print_loss']) == 0 or (i + 1) == len(train_loader)):
            # print(f"rank {parse_args.local_rank} at epoch {e + 1}:{logging_loss}")
            # logging.warning(f"score:{output['score'].clone().detach()}")
            # dist.barrier()
    
            # print(f'reduce')
            dist.all_reduce_multigpu(logging_loss, op=dist.ReduceOp.SUM)
            # print(f"rank {parse_args.local_rank} at epoch {e + 1} sum loss:{logging_loss}")
            logging.warning(f"step {i + 1},loss:{logging_loss / train_args['print_loss']}") 
        
            logging_loss = torch.tensor([0.0], device = device)
        
        logging_loss += loss.clone().detach() / len(valid_loader)
    # checkpoint = {
    #     "bert": model.bert.state_dict(),
    #     "fc": model.linear.state_dict(),
    #     "optimizer": optimizer.state_dict()
    # }
    
    if (parse_args.local_rank == 0):
        torch.save(model.state_dict(), f"{checkpoint_path}/checkpoint_train_{e+1}.pt")
    dist.barrier()
    model.load_state_dict(torch.load(f"{checkpoint_path}/checkpoint_train_{e+1}.pt", map_location=map_location))

    
    print(f'epoch:{e + 1} validation')
    with torch.no_grad():
        eval_loss = torch.tensor([0.0], device = device)
        model.eval()
        for i, data in enumerate(tqdm(valid_loader, ncols = 100)):
            data['input_ids'] = data['input_ids'].to(device)
            data['attention_mask'] = data['attention_mask'].to(device)
            data['labels'] = data['labels'].to(device)
            data['wer'] = data['wer'].to(device)

            output = model(
                    input_ids = data['input_ids'],
                    attention_mask = data['attention_mask'],
                    labels = data['labels']
                )
            loss = output['loss']
            

            if (mode == 'MWER'):
                first_score = data['score'].to(device)

                combined_score = first_score + output['score']

                avg_error = data['avg_error'].to(device)

                # softmax seperately
                index = 0
                for nbest in data['nbest']:

                    combined_score[index: index + nbest] = torch.softmax(
                        combined_score[index:index + nbest], dim = -1
                    )
                    
                    index = index + nbest

                loss_MWER = first_score * (data['wer'] - avg_error)
                loss_MWER = torch.sum(loss_MWER) / torch.tensor(train_args['accumgrad'], dtype = torch.float32)

                loss = loss_MWER + 1e-4 * loss
                
            elif (mode == 'MWED'):
                first_score = data['score'].to(device)
                wer = data['wer'].clone()

                assert(first_score.shape == output['score'].shape), f"first_score:{first_score.shape}, score:{output['score'].shape}"

                combined_score = first_score + output['score'].clone()
                
                index = 0
                scoreSum = torch.tensor([]).to(device)
                werSum = torch.tensor([]).to(device)

                for nbest in data['nbest']:

                    score_sum = torch.sum(
                        combined_score[index: index + nbest].clone()
                    ).repeat(nbest)

                    wer_sum = torch.sum(
                        wer[index:index + nbest].clone()
                    ).repeat(nbest)

                    scoreSum = torch.cat([scoreSum, score_sum])
                    werSum = torch.cat([werSum, wer_sum])

                    index = index + nbest

                index = 0
                
                T = scoreSum / werSum # hyperparameter T
                # print(f'combined_score:{combined_score}')
                combined_score = combined_score / T
                # print(f'combined_score after scale:{combined_score}')

                for nbest in data['nbest']:
                    combined_score[index: index + nbest] = torch.softmax(
                        combined_score[index:index + nbest].clone(), dim = -1
                    )
                    wer[index: index + nbest] = torch.softmax(
                        wer[index:index + nbest].clone(), dim = -1
                    )
                
                    index = index + nbest
                
                # print(f'combined_score after scale & softmax:{combined_score}')
                loss_MWED =  wer * torch.log(combined_score)
                loss_MWED = torch.neg(torch.sum(loss_MWED)) / torch.tensor(train_args['accumgrad'], dtype = torch.float32)
                loss = loss_MWED + 1e-4 * loss
                
            eval_loss += loss.clone().detach() / len(valid_loader)

        dist.all_reduce_multigpu(eval_loss, op=dist.ReduceOp.SUM)

        print(f'epoch:{e + 1}, loss:{eval_loss}')
        logging.warning(f'epoch:{e + 1}, loss:{eval_loss}')

        if (eval_loss < min_val_loss):
            torch.save(model.state_dict(), f"{checkpoint_path}/checkpoint_train_best.pt")
            min_val_loss = eval_loss
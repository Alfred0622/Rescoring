import os
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
from utils.LoadConfig import load_config
from utils.CollateFunc import RescoreBertBatch
from utils.PrepareModel import prepare_RescoreBert
from torch.optim import AdamW

mode = sys.argv[1]

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (len(sys.argv) != 2):
    assert(len(sys.argv) == 2), "python ./train_RescoreBert.py {MD,MWER,MWED}"


use_MWER = False
use_MWED = False
if (mode == "MWER"):
    use_MWER = True
elif (mode == 'MWED'):
    use_MWED = True
else:
    mode = 'MD'

config = f'./config/aishell2_RescoreBert.yaml'
args, train_args, recog_args = load_config(config)

setting = 'withLM' if (args['withLM']) else 'noLM'

if (args['dataset'] in ['aishell2']):
    dev_set = 'dev_ios'
else:
    dev_set = 'dev'

model, tokenizer = prepare_RescoreBert(args['dataset'], device)
model = model.to(device)

if (not os.path.exists(f"./log/RescoreBert/{args['dataset']}/{train_args['mode']}/{setting}")):
    os.makedirs(f"./log/RescoreBert/{args['dataset']}/{train_args['mode']}/{setting}")

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/RescoreBert/{args['dataset']}/{train_args['mode']}/{setting}/train.log",
    filemode="w",
    format=FORMAT,
)

with open(f"./data/{args['dataset']}/{setting}/50best/MLM/train/rescore_data.json") as f, \
     open(f"./data/{args['dataset']}/{setting}/50best/MLM/{dev_set}/rescore_data.json") as dev:
    train_json = json.load(f)
    valid_json = json.load(dev)

train_dataset = getRescoreDataset(train_json, args['dataset'], tokenizer, topk = args['nbest'])
valid_dataset = getRescoreDataset(valid_json, args['dataset'], tokenizer, topk = args['nbest'])

optimizer = AdamW(model.parameters(), lr = float(train_args['lr']))

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size=train_args['train_batch'],
    collate_fn=RescoreBertBatch,
    num_workers = 4
)

valid_loader = DataLoader(
    dataset = valid_dataset,
    batch_size=train_args['valid_batch'],
    collate_fn=RescoreBertBatch,
    num_workers = 4
)

weight = 1e-4

optimizer.zero_grad()
for e in range(train_args['epoch']):
    model.train()

    min_val_loss = 1e8
    min_val_cer = 1e6

    logging_loss = 0.0

    for i, data in enumerate(tqdm(train_loader)):

        data['input_ids'] = data['input_ids'].to(device)
        data['attention_mask'] = data['attention_mask'].to(device)
        data['labels'] = data['labels'].to(device)
        data['wer'] = data['wer'].to(device)
        # data = {k : v.to(device) for k ,v in data.items()}

        output = model(
            input_ids = data['input_ids'],
            attention_mask = data['attention_mask'],
            labels = data['labels']
        )
        loss = output["loss"] / float(train_args['accumgrad'])        

        # MWER
        if (mode == 'MWER'):
            data['score'] = data['score'].to(device)
            combined_score = data['score'] + weight * output['score']
            combined_score = torch.softmax(combined_score)

            avg_error = torch.mean(data['wer'],dim = -1)
            avg_error = avg_error.repeat(combined_score.shape)

            loss_MWER = combined_score * (data['wer'] - avg_error)
            loss_MWER = torch.sum(loss_MWER) / int(train_args['accumgrad'])

            loss = loss_MWER + 1e-4 * loss
        
        elif (mode == 'MWED'):
            data['score'] = data['score'].to(device)
            wer = data['wer']
            combined_score = data['score'] + weight * output['score']

            # wer = wer.reshape(batch_size, -1)
            # weight_sum = weight_sum.reshape(batch_size, -1)
                
            T = torch.sum(combined_score, dim = -1) / torch.sum(wer, dim = -1) # hyperparameter T
            T = T.unsqueeze(-1) # temperature
                
            d_error = torch.softmax(wer, dim=-1)
            d_score = torch.softmax(combined_score / T, dim=-1).reshape(d_error.shape)

            loss_MWED = d_error * torch.log(d_score)
            loss_MWED = torch.neg(torch.sum(loss_MWED)) / int(train_args['accumgrad'])

            loss = loss_MWED + 1e-4 * loss

        loss.backward()

        if ((i + 1) % int(train_args['accumgrad']) == 0):
            optimizer.step()
            optimizer.zero_grad()
        
        if ((i + 1) % int(train_args['print_loss']) == 0 or (i + 1) == len(train_loader)):
            logging.warning(f"score:{output['score'].clone().detach()}")
            logging.warning(f"step {i + 1},loss:{logging_loss / train_args['print_loss']}") 
            logging_loss = 0
        
        logging_loss += loss
    
    checkpoint_path = Path(f"./checkpoint/{args['dataset']}/RescoreBert/{setting}/{mode}/{args['nbest']}best")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "bert": model.bert.state_dict(),
        "fc": model.linear.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_{e+1}.pt")
     
    eval_loss = 0.0
    model.eval()

    print(f'epoch:{e + 1} validation')
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader)):
            data['input_ids'] = data['input_ids'].to(device)
            data['attention_mask'] = data['attention_mask'].to(device)
            data['labels'] = data['labels'].to(device)
            data['wer'] = data['wer'].to(device)

            output = output = model(
                    input_ids = data['input_ids'],
                    attention_mask = data['attention_mask'],
                    labels = data['labels']
                )
            loss = output['loss']
            

            if (mode == 'MWER'):
                combined_score = data['score'] + weight * output['score']
                combined_score = torch.softmax(combined_score)

                avg_error = torch.mean(data['wer'],dim = -1)
                avg_error = avg_error.repeat(combined_score.shape)

                loss_MWER = combined_score * (data['err'] - avg_error)
                loss_MWER = torch.sum(loss_MWER) / int(train_args['accumgrad'])

                loss = loss_MWER + 1e-4 * loss
            
            elif (mode == 'MWED'):
                wer = data['wer']
                combined_score = data['score'] + weight * output['score']

                # wer = wer.reshape(batch_size, -1)
                # weight_sum = weight_sum.reshape(batch_size, -1)
                    
                T = torch.sum(combined_score, dim = -1) / torch.sum(wer, dim = -1) # hyperparameter T
                T = T.unsqueeze(-1)
                    
                d_error = torch.softmax(wer, dim=-1)
                d_score = torch.softmax(combined_score / T, dim=-1).reshape(d_error.shape)

                loss_MWED = d_error * torch.log(d_score)
                loss_MWED = torch.neg(torch.sum(loss_MWED)) / int(train_args['accumgrad'])
                loss = loss_MWED + 1e-4 * loss
            eval_loss += loss

    logging.warning(f'epoch:{e + 1}, loss:{eval_loss / len(valid_loader)}')

    if (eval_loss < min_val_loss):
        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best.pt")
        min_val_loss = eval_loss
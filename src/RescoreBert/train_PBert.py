import os
import sys
sys.path.append("../")
import torch
import logging
import json
import wandb
import math
import random

from tqdm import tqdm
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from src_utils.LoadConfig import load_config
from utils.Datasets import prepareListwiseDataset
from utils.CollateFunc import NBestSampler, BatchSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from src_utils.get_recog_set import get_valid_set
from utils.PrepareModel import preparePBert, prepareContrastBert
from utils.CollateFunc import PBertBatch, PBertBatchWithHardLabel
from utils.PrepareScoring import prepare_score_dict, calculate_cer
# from accelerate import Accelerator

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

checkpoint = None
if (len(sys.argv) >= 2):
    mode = sys.argv[1].upper() # pbert or contrast
    assert (mode in ['PBERT', 'CONTRAST']), "mode must in PBERT or CONTRAST"
    if (len(sys.argv) >= 3):
        checkpoint_path = sys.argv[2]

if (torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if (mode == 'PBERT'):
    config_path = "./config/PBert.yaml"
else:
    config_path = "./config/contrastBert.yaml"

args, train_args, recog_args = load_config(config_path)

setting = 'withLM' if (args['withLM']) else 'noLM'

log_path = f"./log/P_BERT/{args['dataset']}/{setting}/{mode}"
run_name = f"RescoreBert_{mode}_batch{train_args['batch_size']}_lr{train_args['lr']}_Freeze{train_args['freeze_epoch']}"
if (train_args['hard_label']):
    collate_func = PBertBatchWithHardLabel
    run_name = run_name + "_HardLabel_Entropy"
else:
    run_name = run_name + train_args['loss_type']

if ('weightByWER' in train_args.keys() and train_args['weightByWER']  != 'none'):
    run_name = run_name + f"_weightByWER{train_args['weightByWER']}"
    log_path = log_path + "/weightByWER"
else:
    log_path = log_path + "/normal"

log_path = Path(f"./log/RescoreBERT/{args['dataset']}/{setting}/{mode}")
log_path.mkdir(parents = True, exist_ok = True)

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"{log_path}/train_{run_name}.log",
    filemode="w",
    format=FORMAT,
)

valid_set = get_valid_set(args['dataset'])

if (mode == 'PBERT'):
    model, tokenizer = preparePBert(
        args['dataset'], 
        device,
        train_args['hard_label'],
        train_args['weightByWER']
    )
elif (mode == 'CONTRAST'):
    model, tokenizer = prepareContrastBert(args, train_args)

print(type(model))
model = model.to(device)
if (torch.cuda.device_count() > 1):
    model = torch.nn.DataParallel(model)
# optimizer = AdamW(model.parameters(), lr = float(train_args['lr']))
optimizer = Adam(model.parameters(), lr = float(train_args['lr']))

with open(f"../../data/{args['dataset']}/data/{setting}/train/data.json") as f, \
     open(f"../../data/{args['dataset']}/data/{setting}/{valid_set}/data.json") as dev:
    train_json = json.load(f)
    valid_json = json.load(dev)

"""
Load checkpoint
"""
start_epoch = 0

get_num = -1
if ('WANDB_MODE' in os.environ.keys() and os.environ['WANDB_MODE'] == 'disabled'):
    get_num = 550

print(f"tokenizing Train")
train_dataset = prepareListwiseDataset(
    data_json = train_json, 
    dataset = args['dataset'], 
    tokenizer = tokenizer, 
    sort_by_len = True, 
    get_num=get_num
)
print(f"tokenizing Validation")
valid_dataset = prepareListwiseDataset(
    data_json = valid_json, 
    dataset = args['dataset'], 
    tokenizer = tokenizer, 
    sort_by_len = True, 
    get_num=get_num
)

print(f"Prepare Sampler")
train_sampler = NBestSampler(train_dataset)
valid_sampler = NBestSampler(valid_dataset)

print(f"len of sampler:{len(train_sampler)}")

train_batch_sampler = BatchSampler(train_sampler, train_args['batch_size'])
valid_batch_sampler = BatchSampler(valid_sampler, train_args['batch_size'])

print(f"len of batch sampler:{len(train_batch_sampler)}")

collate_func = PBertBatch


train_loader = DataLoader(
    dataset = train_dataset,
    batch_sampler = train_batch_sampler,
    collate_fn = collate_func,
    num_workers=16,
    pin_memory=True
)

valid_loader = DataLoader(
    dataset = valid_dataset,
    batch_sampler = valid_batch_sampler,
    collate_fn = collate_func,
    num_workers=16,
    pin_memory=True
)

warmup_step = int(train_args['warmup_step'])
total_step = len(train_batch_sampler) * int(train_args['epoch'])

print(f"single step : {len(train_batch_sampler)}")
print(f"total steps : {len(train_batch_sampler) * int(train_args['epoch'])}")

# print(warmup_step/total_step)

lr_scheduler = OneCycleLR(
    optimizer, 
    max_lr = float(train_args['lr']) * 10, 
    epochs = int(train_args['epoch']), 
    steps_per_epoch = len(train_batch_sampler),
    pct_start = 0.01
)

index_dict, inverse_dict,am_scores, ctc_scores, lm_scores, rescores, wers, hyps, refs = prepare_score_dict(valid_json, nbest = args['nbest'])

"""
Initialize wandb
"""
config = {
    "args": args,
    "train_args": train_args,
    "Bert_config": model.bert.config if (torch.cuda.device_count() <= 1) else model.module.model.config
}

wandb.init(
    project = f"NBestBert_{args['dataset']}_{setting}",
    config = config, 
    name = run_name
)

checkpoint_path = Path(f"./checkpoint/{args['dataset']}/NBestCrossBert/{setting}/{mode}/{args['nbest']}best/{run_name}")
checkpoint_path.mkdir(parents=True, exist_ok=True)
"""
Start Training
"""

optimizer.zero_grad(set_to_none=True)
wandb.watch(model)

step = 0
min_val_loss = 1e8
min_val_cer = 1e6
logging_loss = torch.tensor([0.0])

for param in model.bert.parameters():
        param.requires_grad = False

# accelerator = Accelerator()
# device = accelerator.device
# model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
#     model, optimizer, train_loader, lr_scheduler
# )

for e in range(start_epoch, train_args['epoch']):
    train_epoch_loss = torch.tensor([0.0])
    model.train()
    if (e >= int(train_args['freeze_epoch'])):
        print('Unfreeze BERT')
        for param in model.bert.parameters():
            param.requires_grad = True
    else:
        print("Freeze BERT")

    for i, data in enumerate(tqdm(train_loader, ncols = 100)):
        # print(f"data:{data['wers']}")
        for key in data.keys():
            if (key not in ['name', 'indexes']):
                # print(f"{key}:{type(data[key])}")
                data[key] = data[key].to(device)
        
        if (mode == "CONTRAST" or ('weightByWER' in train_args.keys() and train_args['weightByWER'] == 'none')):
            data['wers'] = None

        output = model.forward(
            **data
        )

        loss = output['loss']
        loss = torch.mean(loss)
        
        if (torch.cuda.device_count() > 1):
            loss = loss.sum()
        loss.backward()

        if ( ((i + 1) % int(train_args['accumgrad'])) == 0 or (i + 1) == len(train_batch_sampler)):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        
        if ( (step > 0) and (step % int(train_args['print_loss'])) == 0):
            logging_loss = logging_loss / step
            logging.warning( f"epoch:{e + 1} step {i + 1},loss:{logging_loss}" )
            wandb.log(
                {
                    "train_loss": logging_loss
                }, step = (i + 1) + e * len(train_batch_sampler)
            )

            logging_loss = torch.tensor([0.0])
            step = 0
        
        logging_loss += loss.item()
        train_epoch_loss += loss.item()

    if (e == 0 or (e + 1) % 5 == 0 ):    
        checkpoint = {
            "epoch": e,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict()
        }

        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_{e+1}.pt")
    
    wandb.log(
                {
                    "train_loss": train_epoch_loss,
                    "epoch": e + 1
                }, step = (i + 1) + e * len(train_batch_sampler)
            )
    
    """
    Validation
    """
    model.eval()
    valid_len = len(valid_batch_sampler)
    eval_loss = torch.tensor([0.0])
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader, ncols = 100)):
            for key in data.keys():
                if (key not in ['name', 'indexes']):

                    data[key] = data[key].to(device)

            if (mode == "CONTRAST" or ('weightByWER' in train_args.keys() and train_args['weightByWER'] == 'none')):
                data['wers'] = None

            output = model.forward(
                **data
            )

            loss = output['loss']
            scores = output['score']
            loss = torch.mean(loss)
            if (torch.cuda.device_count() > 1):
                loss = loss.sum()
            eval_loss += loss.item()

            """
            Calculate Score
            """
            for n, (name, index, score) in enumerate(zip(data['name'], data['indexes'], scores)):
                rescores[index_dict[name]][index] += score.item()
        
        print(f"Validation: Calcuating CER")
        best_am, best_ctc, best_lm, best_rescore, min_cer = calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            am_range = [0, 1],
            ctc_range = [0, 1],
            lm_range = [0, 1],
            rescore_range = [0, 1],
            search_step = 0.1 ,
            recog_mode = False
        )              

        
        eval_loss = (eval_loss / len(valid_batch_sampler))
        print(f'epoch:{e + 1},Validation loss:{eval_loss}')
        print(f'epoch:{e + 1},Validation CER:{min_cer}, weight = {[best_am, best_ctc, best_lm, best_rescore]}')
        wandb.log(
            {
            "eval_loss": eval_loss,
            "eval_CER": min_cer,
            "epoch": (e + 1)
            },
            step = ( (e + 1) * len(train_batch_sampler) )
        )
        logging.warning(f'epoch:{e + 1},validation loss:{eval_loss}')
        logging.warning(f'epoch:{e + 1},validation CER:{min_cer}, , weight = {[best_am, best_ctc, best_lm, best_rescore]}')

        rescores = np.zeros(rescores.shape, dtype = float)

        if (eval_loss < min_val_loss):
            torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best.pt")
            min_val_loss = eval_loss
        if (min_cer < min_val_cer):
            torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best_CER.pt")
            min_val_cer = min_cer

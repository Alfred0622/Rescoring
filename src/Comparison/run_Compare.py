import json
from tkinter import ON
import yaml
import random
import torch
import glob
import logging
import os
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model.BertForComparison import Bert_Sem, Bert_Compare
from utils.Datasets import(
    get_dataset,
    get_recogDataset    
)
from utils.CollateFunc import(
    bertCompareBatch,
)

from utils.Datasets import get_dataset
from utils.PrepareModel import prepare_model
from src_utils.LoadConfig import load_config
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
import wandb

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

"""Basic setting"""
# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

config = "./config/comparison.yaml"
args, train_args, recog_args = load_config(config)

setting = 'withLM' if args['withLM'] else 'noLM'

if (not os.path.exists(f"./log/{args['dataset']}/{setting}/Bert_sem")):
    os.makedirs(f"./log/{args['dataset']}/{setting}/Bert_sem")

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"

logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/{args['dataset']}/{setting}/Bert_sem/train_batch{train_args['train_batch']}_lr{train_args['lr']}.log",
    filemode="w",
    format=FORMAT,
)

# Prepare Data
print('Data Prepare')
print(f'setting:{setting}')
print(f"nbest:{args['nbest']}")


# if (args['stage'] <= 0) and (args['stop_stage']>= 0):
model, tokenizer = prepare_model(args, train_args, device)
wandb.init()

optimizer = AdamW(model.parameters(), lr = float(train_args['lr']))

if (torch.cuda.device_count() > 1):
    model = torch.nn.DataParallel(model)

if (len(sys.argv) == 2):
    checkpoint = sys.argv[1]

    start_epoch = int(checkpoint.split(".")[0][-1])

    load_checkpoint = torch.load(checkpoint)

    print(f"load checkpoint from: {checkpoint}")

    model.bert.load_state_dict(load_checkpoint['state_dict'])
    model.linear.load_state_dict(load_checkpoint['fc_checkpoint'])
    optimizer.load_state_dict(load_checkpoint['optimizer'])
    

else: start_epoch = 0

print(f'training')
min_loss = 1e8
loss_seq = []

train_path = f"./data/{args['dataset']}/train/{setting}/{args['nbest']}best/data.json"
valid_path = f"./data/{args['dataset']}/valid/{setting}/5best/data.json"

with open(train_path, 'r') as f ,\
    open(valid_path, 'r') as v:
    train_json = json.load(f)   #[:train_args['train_batch'] * 512]
    valid_json = json.load(v)
print(f"# of train data:{len(train_json)}")
print(f"# of valid data:{len(valid_json)}")

print(f'tokenizing data......')
valid_dataset, valid_json = get_dataset(valid_json, tokenizer)
# with open(f"./data/{args['dataset']}/valid/{setting}/{args['nbest']}best/token.json", 'w') as valid:
#     json.dump(valid_json, valid, ensure_ascii = False, indent = 4)

train_dataset, train_json = get_dataset(train_json, tokenizer)
# with open(f"./data/{args['dataset']}/train/{setting}/{args['nbest']}best/token.json", 'w') as train:
#     json.dump(train_json, train, ensure_ascii = False, indent = 4)

train_loader = DataLoader(
    train_dataset,
    batch_size = train_args["train_batch"],
    collate_fn=bertCompareBatch,
    num_workers=8,
    shuffle = True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size = train_args["train_batch"],
    collate_fn=bertCompareBatch,
    num_workers=8,
)

# scheduler = OneCycleLR(
#     model.optimizer, 
#     max_lr = float(train_args['lr']), 
#     steps_per_epoch = len(train_loader),
#     epochs = train_args['epoch'],
#     pct_start = 1 / train_args['epoch'],
#     anneal_strategy = 'linear'
# )

for e in range(start_epoch, train_args["epoch"]):
    model.train()
    if (e < 2):
        print(f'freeze')
        logging.warning(f'freeze')
        if (torch.cuda.device_count() > 1):
            for param in model.module.bert.parameters():
                param.require_grad = False
        else:
            for param in model.bert.parameters():
                param.require_grad = False
    else:
        if (torch.cuda.device_count() > 1):
            for param in model.module.bert.parameters():
                param.require_grad = True
        else:
            for param in model.bert.parameters():
                param.require_grad = True

    for n, data in enumerate(tqdm(train_loader, ncols = 100)):
        
        optimizer.zero_grad()
        logging_loss = 0.0
        # logging.warning(data['labels'])

        data = {k: v.to(device) for k, v in data.items()}
    
        loss = model(**data).loss
        loss = loss.sum() / train_args["accumgrad"]
        loss.backward()
        
        logging_loss += loss.item()

        if ((n + 1) % train_args["accumgrad"] == 0) or ((n + 1) == len(train_loader)):
            optimizer.step()
            # lrs.append(model.optimizer.param_groups[0]["lr"])
            # scheduler.step()
            
        if ((n + 1) % train_args["print_loss"] == 0) or ((n + 1) == len(train_loader)):
            logging.warning(
                f"Training epoch :{e + 1} step:{n + 1}, loss:{logging_loss}"
            )
            loss_seq.append(logging_loss)
            logging_loss = 0.0

    train_checkpoint = dict()
    train_checkpoint["state_dict"] = model.bert.state_dict() if torch.cuda.device_count() == 1 else  model.module.bert.state_dict()
    train_checkpoint["fc_checkpoint"] = model.linear.state_dict() if torch.cuda.device_count() == 1 else  model.module.linear.state_dict()
    train_checkpoint["optimizer"] = optimizer.state_dict()  # if torch.cuda.device_count() > 1 else  model.module.optimizer.state_dict()
    # train_checkpoint["scheduler"] = scheduler.state_dict()
    if (not os.path.exists(f"./checkpoint/{args['dataset']}/{setting}/{args['nbest']}/batch{train_args['train_batch']}_lr{train_args['lr']}")):
        os.makedirs(f"./checkpoint/{args['dataset']}/{setting}/{args['nbest']}/batch{train_args['train_batch']}_lr{train_args['lr']}")
    
    torch.save(
        train_checkpoint,
        f"./checkpoint/{args['dataset']}/{setting}/{args['nbest']}/batch{train_args['train_batch']}_lr{train_args['lr']}/checkpoint_train_{e + 1}.pt",
    )

    # eval
    model.eval()
    valid_loss = 0.0   
    with torch.no_grad():
        for n, data in enumerate(tqdm(valid_loader, ncols = 80)):
            data = {k: v.to(device) for k,v in data.items()}
            loss = model(**data).loss


            valid_loss += loss.sum().item()

    logging.warning(f'epoch:{e + 1} validation loss:{valid_loss}')
    
    if (valid_loss < min_loss):
        torch.save(
            train_checkpoint,
            f"./checkpoint/{args['dataset']}/{setting}/{args['nbest']}/batch{train_args['train_batch']}_lr{train_args['lr']}/checkpoint_train_best.pt",
        )

        min_loss = valid_loss

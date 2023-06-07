import json
import random
import torch
import logging
import os
import sys
sys.path.append("../")
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from model.BertForComparison import Bert_Alsem
from utils.Datasets import(
    get_alsemDataset,
)
from utils.CollateFunc import(
    bertAlsemBatch,
)
from src_utils.LoadConfig import load_config
from utils.PrepareModel import prepare_model
import wandb
from transformers import Trainer, TrainingArguments, DataCollator
from torch.optim.lr_scheduler import OneCycleLR

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

config = f'./config/Bert_alsem.yaml'
args, train_args, recog_args = load_config(config)
dataset = args['dataset']
setting = 'withLM' if args['withLM'] else "noLM"
topk = args['nBest']


log_path = Path(f"./log/{args['dataset']}/{setting}/Bert_alsem")
log_path.mkdir(exist_ok = True, parents = True)

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"

logging.basicConfig(
    level=logging.INFO,
    filename=f"{log_path}/Bert_alsem_train_batch{train_args['train_batch']}_lr{train_args['lr']}.log",
    filemode="w",
    format=FORMAT,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

with open(f"./data/{dataset}/train/{setting}/{args['nBest']}best/data.json") as f:
    train_json = json.load(f)
with open(f"./data/{dataset}/valid/{setting}/5best/data.json") as v:
    valid_json = json.load(v)

model, tokenizer = prepare_model(args, train_args, device)

train_dataset = get_alsemDataset(train_json, args['dataset'], tokenizer)
valid_dataset = get_alsemDataset(valid_json, args['dataset'], tokenizer)

train_loader = DataLoader(
     train_dataset,
     batch_size=train_args['train_batch'],
     collate_fn=bertAlsemBatch,
     num_workers=8
     )

valid_loader = DataLoader(
      valid_dataset,
      batch_size=train_args['valid_batch'],
      collate_fn=bertAlsemBatch,
      num_workers=8
      )

min_val = 1e8

wandb_config = wandb.config
wandb_config = model.bert.config if (torch.cuda.device_count() == 1) else model.module.bert.config
wandb_config.learning_rate = float(train_args['lr'])
wandb_config.batch_size = train_args['train_batch']
cal_batch = int(train_args['train_batch']) * train_args['accum_grads']

scheduler = OneCycleLR(
    model.optimizer, 
    max_lr = float(train_args['lr']), 
    steps_per_epoch = int(len(train_loader) / train_args['accum_grads']) + 1,
    epochs = train_args['epoch'],
    pct_start = 0.02,
    anneal_strategy = 'linear'
)

wandb.init( 
    project = f"Bertsem_{args['dataset']}",
    config = wandb_config, 
    name = f"batch{train_args['train_batch']}_accum{train_args['accum_grads']}_lr{train_args['lr']}"
)
model.optimizer.zero_grad(set_to_none=True)

step = 0
for e in range(train_args['epoch']):
    model.train()
    train_loss = 0.0
    accum_loss = 0.0
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

    model.optimizer.zero_grad(set_to_none=True)
    for i, data in enumerate(tqdm(train_loader, ncols = 100)):
        with torch.autograd.set_detect_anomaly(True):
            data = {k: v.to(device) for k, v in data.items()}
            loss = model(**data).loss
            accum_loss += loss.item()
            train_loss += loss.item()

            loss /= train_args['accum_grads']
            loss.backward()

        if ((i + 1) % train_args['accum_grads'] == 0):
            model.optimizer.step()
            scheduler.step()
            model.optimizer.zero_grad(set_to_none=True)

            step += 1

        if ((step > 0) and step % train_args['print_loss'] == 0):
            wandb.log(
                {"loss": accum_loss},
                step = (i+1) + e * len(train_loader)
            )

            logging.warning(
                f"train_epoch:{e + 1}, step:{i + 1}, loss:{accum_loss / train_args['print_loss']}")
            accum_loss = 0.0


    checkpoint = {
        "epoch": e + 1,
        "bert": model.bert.state_dict(),
        "rnn": model.rnn.state_dict(),
        "fc1": model.fc1.state_dict(),
        "fc2": model.fc2.state_dict(),
        "optimizer": model.optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }

    savePath = Path(f"./checkpoint/{args['dataset']}/{setting}/Bert_alsem_batch{train_args['train_batch']}_lr{train_args['lr']}/")
    savePath.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, f"{savePath}/checkpoint_train_{e + 1}.pt")

    accum_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader, ncols = 100)):
            data = {k: v.to(device) for k, v in data.items()}
            loss = model(**data).loss
            accum_loss += loss.item()

        if (accum_loss < min_val):
            torch.save(checkpoint, f"{savePath}/chechpoint_train_best.pt")
        
        wandb.log(
            {
                "train_epoch_loss": train_loss,
                "valid_loss":accum_loss,
                "epoch": e + 1
            },
            step = len(train_loader) * (e + 1)
        )
        
        logging.warning(f"epoch:{e + 1}, validation loss:{accum_loss}")

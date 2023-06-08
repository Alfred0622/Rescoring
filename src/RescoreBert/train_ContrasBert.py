import os
import sys
import logging
sys.path.append("../")

import random
import numpy as np
import json
import torch
import torch.nn as nn

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
from pathlib import Path

import wandb
from src_utils.get_recog_set import get_valid_set
from utils.PrepareModel import prepareContrastBert
from utils.Datasets import prepareListwiseDataset
from utils.CollateFunc import NBestSampler, BatchSampler, PBertBatchWithHardLabel
from utils.LoadConfig import load_config

config_name = './config/contrastBert.yaml'
args, train_args, recog_args = load_config(config_name)
setting = 'withLM' if args['withLM'] else 'noLM'
mode = 'contrast'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_path = f"./log/contrastBERT/{args['dataset']}/{setting}/{mode}"
run_name = f"RescoreBert_{mode}_batch{train_args['batch_size']}_lr{train_args['lr']}_Freeze{train_args['freeze_epoch']}"
log_path.mkdir(parents = True, exist_ok = True)

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"{log_path}/train_{run_name}.log",
    filemode="w",
    format=FORMAT,
)

model, tokenizer = prepareContrastBert(
    args,
    train_args
)

model.train()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = float(train_args['lr'])) 

valid_set = get_valid_set(args['dataset'])

train_path = f"../../data/{args['dataset']}/data/{setting}/train/data.json"
dev_path = f"../../data/{args['dataset']}/data/{setting}/{valid_set}/data.json"
if (len(sys.argv) >= 2):
    file_path = sys.argv[1]
    train_path = f"{file_path}/data/{args['dataset']}/data/{setting}/train/data.json"
    dev_path = f"{file_path}/data/{args['dataset']}/data/{setting}/{valid_set}/data.json"

with open(train_path) as train, open(dev_path) as dev:
    train_json = json.load(train)
    valid_json = json.load(dev)

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

train_sampler = NBestSampler(train_dataset)
valid_sampler = NBestSampler(valid_dataset)

print(f"len of sampler:{len(train_sampler)}")

train_batch_sampler = BatchSampler(
    train_sampler, 
    train_args['batch_size']
)
valid_batch_sampler = BatchSampler(
    valid_sampler, 
    train_args['batch_size']
)

print(f"len of batch sampler:{len(train_batch_sampler)}")
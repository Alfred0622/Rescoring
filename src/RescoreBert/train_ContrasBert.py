import os
import sys
import logging
sys.path.append("../")

import random
import numpy as np
import torch
import torch.nn as nn

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
from pathlib import Path

import wandb

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


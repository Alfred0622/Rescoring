import json
import yaml
import random
import torch
import glob
import logging
import os
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
from utils.LoadConfig import load_config
from utils.PrepareModel import prepare_model

from transformers import Trainer, TrainingArguments, DataCollator

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

config = f'./config/Bert_alsem.yaml'
args, train_args, recog_args = load_config(config)
dataset = args['dataset']
setting = 'withLM' if args['withLM'] else "noLM"
topk = args['nBest']

if (not os.path.exists(f"./log/{setting}/Bert_alsem")):
    os.makedirs(f"./log/{setting}/Bert_alsem")

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"

logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/{setting}/Bert_alsem/Bert_alsem_train.log",
    filemode="w",
    format=FORMAT,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

with open(f"./data/{dataset}/train/{setting}/{args['nBest']}best/data.json") as f:
    train_json = json.load(f)
with open(f"./data/{dataset}/valid/{setting}/{args['nBest']}best/data.json") as v:
    valid_json = json.load(v)

model, tokenizer = prepare_model(args, train_args, device)

train_dataset = get_alsemDataset(train_json, tokenizer)
valid_dataset = get_alsemDataset(valid_json, tokenizer)

train_loader = DataLoader(
     train_dataset,
     batch_size=train_args['train_batch'],
     collate_fn=bertAlsemBatch,
     )

valid_loader = DataLoader(
      valid_dataset,
      batch_size=train_args['valid_batch'],
      collate_fn=bertAlsemBatch,
      )

min_val = 1e8

for e in range(train_args['epoch']):
    model.train()
    if (e < 2):
        for param in model.bert.parameters():
            param.require_grads = False
    accum_loss = 0.0
    model.optimizer.zero_grad()
    for i, data in enumerate(tqdm(train_loader)):
        data = {k: v.to(device) for k, v in data.items()}
        loss = model(**data).loss
        accum_loss += loss.item()
        loss /= train_args['accum_grads']
        loss.backward()

        if ((i + 1) % train_args['accum_grads'] == 0):
            model.optimizer.step()
            model.optimizer.zero_grad()

        if ((i + 1) % train_args['print_loss'] == 0):
            logging.warning(
                f"train_epoch:{e}, step:{i + 1}, loss:{accum_loss / train_args['print_loss']}")
            accum_loss = 0.0

    checkpoint = {
        "epoch": e + 1,
        "bert": model.bert.state_dict(),
        "rnn": model.rnn.state_dict(),
        "fc1": model.fc1.state_dict(),
        "fc2": model.fc2.state_dict(),
        "optimizer": model.optimizer.state_dict()
    }

    savePath = Path(f"./checkpoint/{setting}/Bert_alsem/")
    savePath.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, f"{savePath}/checkpoint_train_{e + 1}.pt")

    accum_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader)):
            data = {k: v.to(device) for k, v in data.items()}
            loss = model(**data).loss
            accum_loss += loss.item()

        if (accum_loss < min_val):
            torch.save(checkpoint, f"{savePath}/chechpoint_train_best.pt")

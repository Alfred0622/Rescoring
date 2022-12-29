import json
import logging
import torch
import os

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.Datasets import get_dataset
from utils.LoadConfig import load_config
from utils.CollateFunc import trainBatch
from utils.PrepareModel import prepare_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

config = f'./config/classification.yaml'

args, train_args, recog_args = load_config(config)
setting = 'withLM' if args['withLM'] else 'noLM'



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, tokenizer = prepare_model(args, train_args, device)
model = model.to(device)

# training
if (args['stage'] <= 0):
    with open(f"../../data/{args['dataset']}/data/{setting}/train/data.json") as f,\
     open(f"../../data/{args['dataset']}/data/{setting}/dev/data.json") as d:
        train_json = json.load(f)
        valid_json = json.load(d)

    train_dataset = get_dataset(train_json, tokenizer, nbest = args['nbest'], for_train = True)
    valid_dataset = get_dataset(valid_json, tokenizer, nbest = args['nbest'], for_train = True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_args['train_batch'],
        collate_fn=trainBatch,
        num_workers=1
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=train_args['valid_batch'],
        collate_fn=trainBatch,
        num_workers=1
    )

    optimizer = AdamW(
        model.parameters(),
        lr = float(train_args['lr'])
    )

    # scheduler = OneCycleLR(
    #     optimizer = optimizer,
    #     total_steps = len(train_loader) * train_args['epoch'],
    #     pct_start = 0.1,
    #     max_lr = float(train_args['lr'])
    # )

    min_loss = 1e9

    for e in range(train_args['epoch']):
        model.train()
        logging_loss = 0.0
        optimizer.zero_grad()
        for step, data in enumerate(tqdm(train_loader)):
            data = {k: v.to(device) for k, v in data.items()}
            loss = model(**data).loss

            loss = loss / train_args['accum_grad']

            logging_loss += loss.item()

            if (step + 1 % train_args['accum_grad'] == 0):
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()

            if (step + 1 % train_args['logging_step'] == 0 or step + 1 == len(train_loader)):
                logging.warning(f"step:{step + 1}: loss:{logging_loss}")
                logging_loss = 0.0
        checkpoint = {
            "epoch": e + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "scheduler": scheduler.state_dict()
        }

        checkpoint_path = Path(f"./checkpoint/{args['dataset']}/{setting}")
        checkpoint_path.mkdir(parents = True, exist_ok= True)

        torch.save(checkpoint, f"{checkpoint_path}/checkpoint.train.{e + 1}.pt")

        
        valid_loss = 0.0
        for data in tqdm(valid_loader):
            data = {k: v.to(device) for k, v in data.items()}
            loss = model(**data).loss

            valid_loss += loss.item()

        logging.warning(f'step:{e + 1}, validation_loss = {valid_loss}')

        if (valid_loss < min_loss):
            min_loss = valid_loss
            torch.save(checkpoint, f"{checkpoint_path}/checkpoint.train.best.pt")

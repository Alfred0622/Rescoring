import os
import sys

sys.path.append("../")
import torch
import logging
import json
import wandb
import math
import random

import gc
from tqdm import tqdm
from pathlib import Path
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from src_utils.LoadConfig import load_config
from utils.Datasets import preparePretrainDataset
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR
from src_utils.get_recog_set import get_valid_set
from utils.PrepareModel import preparePBert, prepareContrastBert
from utils.CollateFunc import pretrainBatch
from utils.PrepareScoring import prepare_score_dict, calculate_cer, get_result
import gc

# from accelerate import Accelerator

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args, train_args, recog_args = load_config("./config/contrastBert.yaml")
setting = "withLM" if args["withLM"] else "noLM"

config = {"args": args, "train_args": train_args}

model, tokenizer = prepareContrastBert(args, train_args)

model = model.to(device)
for params in model.bert.parameters():
    params.requires_frad = False

# load_data
with open(
    f"./data/{args['dataset']}/{setting}/train/pretrain_data.json"
) as train, open(f"./data/{args['dataset']}/{setting}/dev/pretrain_data.json") as valid:
    train_json = json.load(train)
    valid_json = json.load(valid)

if "WANDB_MODE" in os.environ.keys() and os.environ["WANDB_MODE"] == "disabled":
    get_num = 200
else:
    get_num = -1

train_dataset = preparePretrainDataset(
    train_json, dataset=args["dataset"], tokenizer=tokenizer, get_num=get_num
)

valid_dataset = preparePretrainDataset(
    valid_json, dataset=args["dataset"], tokenizer=tokenizer, get_num=get_num
)

train_loader = DataLoader(
    train_dataset,
    batch_size=int(train_args["batch_size"]),
    collate_fn=pretrainBatch,
    num_workers=16,
    shuffle=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=int(train_args["batch_size"]),
    collate_fn=pretrainBatch,
    num_workers=16,
    shuffle=True,
)

optimizer = AdamW(model.parameters(), lr=float(train_args["lr"]))

scheduler = OneCycleLR(
    optimizer,
    max_lr=float(train_args["lr"]),
    pct_start=0.1,
    steps_per_epoch=len(train_loader) // int(train_args["accumgrad"]),
    epochs=10,
)

# pretrain

add_classify = True
run_name = f"pretrain_LSTM_LM_lr{train_args['lr']}_warmup{0.1}"
if add_classify:
    run_name = run_name + "_addClassier"
checkpoint_path = Path(
    f"./checkpoint/{args['dataset']}/NBestCrossBert/{setting}/CONTRAST/{run_name}/{args['nbest']}best/{run_name}"
)
checkpoint_path.mkdir(exist_ok=True, parents=True)

wandb.init(project=f"myBert_{args['dataset']}_{setting}", config=config, name=run_name)
wandb.watch(model)
model.train()
optimizer.zero_grad()
step = 0

logging_loss = torch.tensor(0.0)
logging_lm_loss = torch.tensor(0.0)
logging_bce_loss = torch.tensor(0.0)
min_loss = 1e9


for e in range(10):
    if e < 3:
        print(f"freeze")
        for param in model.LSTMLMHead.parameters():
            param.requires_grad = False
        for param in model.pretrain_classifier.parameters():
            param.requires_grad = False
    else:
        for param in model.LSTMLMHead.parameters():
            param.requires_grad = True
        for param in model.pretrain_classifier.parameters():
            param.requires_grad = True

    train_epoch_loss = torch.tensor(0.0)
    epoch_lm_loss = torch.tensor(0.0)
    epoch_bce_loss = torch.tensor(0.0)

    model.train()
    for i, data in enumerate(tqdm(train_loader)):
        for key in data.keys():
            data[key] = data[key].to(device)

        train_loss = model.pretrain(**data, add_classify=add_classify)

        loss = train_loss["loss"] / int(train_args["accumgrad"])
        loss.backward()
        logging_loss += loss.item()
        train_epoch_loss += loss.item()

        logging_lm_loss += train_loss["LM_loss"].item() / int(train_args["accumgrad"])
        epoch_lm_loss += train_loss["LM_loss"].item() / int(train_args["accumgrad"])

        if train_loss["classifier_loss"] is not None:
            logging_bce_loss += train_loss["classifier_loss"].item() / int(
                train_args["accumgrad"]
            )
            epoch_bce_loss += train_loss["classifier_loss"].item() / int(
                train_args["accumgrad"]
            )

        if (i + 1) % train_args["accumgrad"] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

        if (step > 0) and step % train_args["print_loss"] == 0:
            log_dict = {
                "train_loss": (logging_loss / step),
                "train_lm_loss": (logging_lm_loss / step),
            }
            if add_classify:
                log_dict["train_bce_loss"] = logging_bce_loss / step
            wandb.log(log_dict, step=(e * len(train_loader) + i))

            logging_loss = torch.tensor(0.0)
            logging_lm_loss = torch.tensor(0.0)
            logging_bce_loss = torch.tensor(0.0)

    # validation
    model.eval()
    val_loss = torch.tensor(0.0)
    valid_lm_loss = torch.tensor(0.0)
    valid_bce_loss = torch.tensor(0.0)

    for i, data in enumerate(tqdm(train_loader)):
        for key in data.keys():
            data[key] = data[key].to(device)

        with torch.no_grad():
            train_loss = model.pretrain(**data)
            loss = train_loss["loss"]
            val_loss += loss.item()

            valid_lm_loss += train_loss["LM_loss"].item()
            if train_loss["classifier_loss"] is not None:
                valid_bce_loss += train_loss["classifier_loss"].item()

    log_dict = {
        "train_epoch_loss": train_epoch_loss
        / (len(train_loader) / int(train_args["accumgrad"])),
        "LM_epoch_loss": epoch_lm_loss
        / (len(train_loader) / int(train_args["accumgrad"])),
        "val_loss": val_loss / len(valid_loader),
        "valid_lm_loss": valid_lm_loss / len(valid_loader),
        "epoch": e + 1,
    }

    if add_classify:
        log_dict["bce_epoch_loss"]: epoch_bce_loss / (
            len(train_loader) / int(train_args["accumgrad"])
        )
        log_dict["valid_bce_loss"]: valid_bce_loss / len(valid_loader)
    wandb.log(
        log_dict,
        step=(e + 1) * len(train_loader),
    )

    print(
        f"epoch:{e + 1} , train_loss:{train_epoch_loss / (len(train_loader) / int(train_args['accumgrad']))}, valid_loss:{val_loss / len(valid_loader)}"
    )

    if e % 1 == 0:
        checkpoint = {
            "epoch": e,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_Pretrain_{e+1}.pt")

    if val_loss < min_loss:
        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_Pretrain_BestLoss.pt")
        min_loss = val_loss

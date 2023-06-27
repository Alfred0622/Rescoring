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
from utils.Datasets import prepareListwiseDataset
from utils.CollateFunc import NBestSampler, BatchSampler
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR
from src_utils.get_recog_set import get_valid_set
from utils.PrepareModel import preparePBert, prepareContrastBert, prepareFuseBert
from utils.CollateFunc import PBertBatch, PBertBatchWithHardLabel
from utils.PrepareScoring import prepare_score_dict, calculate_cer
import gc

# from accelerate import Accelerator

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

checkpoint = None
assert len(sys.argv) >= 2, "need to input mode"
if len(sys.argv) >= 2:
    mode = sys.argv[1].upper().strip()  # pbert or contrast

    if len(sys.argv) >= 3:
        checkpoint_path = sys.argv[2]

assert mode in [
    "PBERT",
    "CONTRAST",
    "MARGIN",
    "MARGIN_TORCH",
    "N-FUSE",
], "mode must in PBERT, MARGIN, MARGIN_TORCH, N-FUSE or CONTRAST"

print(f"mode:{mode}")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if mode in ["PBERT", "N-FUSE"]:
    config_path = "./config/PBert.yaml"
else:
    config_path = "./config/contrastBert.yaml"

args, train_args, recog_args = load_config(config_path)

print(f"nBest:{args['nbest']}")
setting = "withLM" if (args["withLM"]) else "noLM"

print(f"batch:{train_args['batch_size']}")

log_path = f"./log/P_BERT/{args['dataset']}/{setting}/{mode}"
run_name = f"TWCC_{mode}_{args['nbest']}Best_batch{train_args['batch_size']}_lr{train_args['lr']}_Freeze{train_args['freeze_epoch']}"
if train_args["hard_label"]:
    collate_func = PBertBatchWithHardLabel
    run_name = run_name + f"_HardLabel_{train_args['loss_type']}"
else:
    run_name = run_name + train_args["loss_type"]

if "weightByWER" in train_args.keys() and train_args["weightByWER"] != "none":
    run_name = run_name + f"_weightByWER{train_args['weightByWER']}"
    log_path = log_path + "/weightByWER"
else:
    log_path = log_path + "/normal"


if train_args["layer_op"] is not None:
    run_name += f"_{train_args['layer_op']}"
else:
    print(f"=============== No Layer Operation ===============")

if mode in ["MARGIN", "MARGIN_TORCH"]:
    run_name = run_name + f"_{mode}_{train_args['margin_value']}"
    if "margin_mode" in train_args.keys() and train_args["margin_mode"] is not None:
        run_name = (
            run_name
            + f"_converge{train_args['converge']}"
            + f"_MarginFirst{train_args['margin_mode']}"
        )
    if "useTopOnly" in train_args.keys() and train_args["useTopOnly"]:
        run_name = run_name + f"_useTopOnly"

elif mode == "CONTRAST":
    run_name = run_name + f"_contrastWeight{train_args['contrast_weight']}"
    run_name += f"_{train_args['compareWith']}"
    if train_args["useTopOnly"]:
        run_name += "_useTopOnly"

    elif train_args["compareWith"] == "POOL":
        # run_name += "_contrastWithPooling"
        if train_args["noCLS"]:
            run_name += "_noCLS"
        if train_args["noSEP"]:
            run_name += "_noSEP"


log_path = Path(f"./log/RescoreBERT/{args['dataset']}/{setting}/{mode}")
log_path.mkdir(parents=True, exist_ok=True)

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"{log_path}/train_{run_name}.log",
    filemode="w",
    format=FORMAT,
)

valid_set = get_valid_set(args["dataset"])

if mode == "PBERT":
    model, tokenizer = preparePBert(args, train_args, device)
elif mode == "CONTRAST" or mode in ["MARGIN", "MARGIN_TORCH"]:
    model, tokenizer = prepareContrastBert(args, train_args, mode)

elif mode == "N-FUSE":
    model, tokenizer = prepareFuseBert(args, train_args)

print(type(model))
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
optimizer = AdamW(model.parameters(), lr=float(train_args["lr"]))
# optimizer = Adam(model.parameters(), lr=float(train_args["lr"]))

if "margin_mode" in train_args.keys():
    print(f"margina_first:{train_args['margin_mode']}")
    if train_args["margin_mode"] is not None:
        run_name += f"_warmup-{train_args['margin_mode']}"

print(f"Run_Name:{run_name}")

with open(f"../../data/{args['dataset']}/data/{setting}/train/data.json") as f, open(
    f"../../data/{args['dataset']}/data/{setting}/{valid_set}/data.json"
) as dev:
    train_json = json.load(f)
    valid_json = json.load(dev)

"""
Load checkpoint
"""
start_epoch = 0
print(f"\n train_args:{train_args} \n")


get_num = -1
save_checkpoint = True
if "WANDB_MODE" in os.environ.keys() and os.environ["WANDB_MODE"] == "disabled":
    get_num = 2000
    save_checkpoint = False
print(f"tokenizing Train")
train_dataset = prepareListwiseDataset(
    data_json=train_json,
    dataset=args["dataset"],
    tokenizer=tokenizer,
    sort_by_len=True,
    get_num=get_num,
    paddingNBest=(mode == "N-FUSE"),
    topk=int(args["nbest"]),
)
print(f"tokenizing Validation")
valid_dataset = prepareListwiseDataset(
    data_json=valid_json,
    dataset=args["dataset"],
    tokenizer=tokenizer,
    sort_by_len=True,
    get_num=get_num,
    paddingNBest=(mode == "N-FUSE"),
    topk=int(args["nbest"]),
)

print(f"Prepare Sampler")
train_sampler = NBestSampler(train_dataset)
valid_sampler = NBestSampler(valid_dataset)

print(f"len of sampler:{len(train_sampler)}")

train_batch_sampler = BatchSampler(train_sampler, train_args["batch_size"])
valid_batch_sampler = BatchSampler(valid_sampler, train_args["batch_size"])

print(f"len of batch sampler:{len(train_batch_sampler)}")

train_collate_func = PBertBatch
valid_collate_func = PBertBatch
if train_args["hard_label"]:
    train_collate_func = partial(PBertBatchWithHardLabel, use_Margin=(mode == "MARGIN"))
    valid_collate_func = partial(PBertBatchWithHardLabel, use_Margin=False)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=train_collate_func,
    num_workers=16,
    pin_memory=True,
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_sampler=valid_batch_sampler,
    collate_fn=valid_collate_func,
    num_workers=16,
    pin_memory=True,
)

# warmup_step = int(train_args["warmup_step"])
total_step = len(train_batch_sampler) * int(train_args["epoch"])

print(f"single step : {len(train_batch_sampler)}")
print(f"total steps : {len(train_batch_sampler) * int(train_args['epoch'])}")

# print(warmup_step/total_step)

lr_scheduler = OneCycleLR(
    optimizer,
    max_lr=float(train_args["lr"]) * 10,
    epochs=int(train_args["epoch"]),
    steps_per_epoch=len(train_batch_sampler),
    pct_start=float(train_args["warmup_ratio"]),
)

(
    index_dict,
    inverse_dict,
    am_scores,
    ctc_scores,
    lm_scores,
    rescores,
    wers,
    hyps,
    refs,
) = prepare_score_dict(valid_json, nbest=args["nbest"])

"""
Initialize wandb
"""
config = {
    "args": args,
    "train_args": train_args,
    "Bert_config": model.bert.config
    if (torch.cuda.device_count() <= 1)
    else model.module.bert.config,
}

wandb.init(project=f"myBert_{args['dataset']}_{setting}", config=config, name=run_name)

checkpoint_path = Path(
    f"./checkpoint/{args['dataset']}/NBestCrossBert/{setting}/{mode}/{args['nbest']}best/{run_name}"
)
checkpoint_path.mkdir(parents=True, exist_ok=True)
del train_json
del valid_json
gc.collect()

"""
Start Training
"""

optimizer.zero_grad(set_to_none=True)
wandb.watch(model)

step = 0
min_val_loss = 1e8
min_val_cer = 1e6
last_val_cer = 1e6

logging_loss = torch.tensor([0.0])
logging_CE_loss = torch.tensor([0.0])
logging_contrastive_loss = torch.tensor([0.0])

for param in model.bert.parameters():
    param.requires_grad = False

# accelerator = Accelerator()
# device = accelerator.device
# model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
#     model, optimizer, train_loader, lr_scheduler
# )
if mode == ["MARGIN", "CONTRAST"] and train_args["margin_mode"] is not None:
    extra_warmup = train_args["margin_mode"] in ["epoch", "WER"]
elif mode in ["MARGIN", "MARGIN_TORCH", "CONTRAST"]:
    extra_warmup = True
else:
    extra_warmup = False

for e in range(start_epoch, train_args["epoch"]):
    train_epoch_loss = torch.tensor([0.0])
    epoch_CE_loss = torch.tensor([0.0])
    epoch_contrast_loss = torch.tensor([0.0])
    model.train()
    if e >= int(train_args["freeze_epoch"]):
        print("Unfreeze BERT")
        for param in model.bert.parameters():
            param.requires_grad = True
    else:
        print("Freeze BERT")

    if (
        "margin_mode" in train_args.keys()
        and e >= 1
        and train_args["margin_mode"] == "epoch"
    ):

        extra_warmup = False
    print(f"extra_warmup = {extra_warmup}")
    for i, data in enumerate(tqdm(train_loader, ncols=100)):
        # for rank in data['wer_rank']:
        #     if (len(rank) < 50):
        #         print(f'filtered:{rank}')
        for key in data.keys():
            if key not in ["name", "indexes", "wer_rank"] and data[key] is not None:
                # print(f"{key}:{type(data[key])}")
                data[key] = data[key].to(device)

        if mode in ["CONTRAST", "MARGIN"] or (
            "weightByWER" in train_args.keys() and train_args["weightByWER"] == "none"
        ):
            data["wers"] = None

        output = model.forward(
            **data,
            extra_loss=extra_warmup,
        )

        loss = output["loss"]
        loss = torch.mean(loss)

        if mode in ["CONTRAST", "MARGIN"] and output["contrast_loss"] is not None:
            epoch_CE_loss += output["CE_loss"].sum().item()
            epoch_contrast_loss += output["contrast_loss"].sum().item()

        if torch.cuda.device_count() > 1:
            loss = loss.sum()
        loss.backward()

        if ((i + 1) % int(train_args["accumgrad"])) == 0 or (i + 1) == len(
            train_batch_sampler
        ):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

        if (step > 0) and (step % int(train_args["print_loss"])) == 0:
            logging_loss = logging_loss / step
            logging.warning(f"epoch:{e + 1} step {i + 1},loss:{logging_loss}")
            wandb.log(
                {"train_loss": logging_loss},
                step=(i + 1) + e * len(train_batch_sampler),
            )

            logging_loss = torch.tensor([0.0])
            step = 0

        logging_loss += loss.item()
        train_epoch_loss += loss.item()

    if save_checkpoint and (e == 0 or (e + 1) % 5 == 0):
        checkpoint = {
            "epoch": e,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
        }

        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_{e+1}.pt")

    if mode in ["CONTRAST", "MARGIN"] and output["contrast_loss"] is not None:
        wandb.log(
            {
                "train_loss": train_epoch_loss,
                "CE_loss": epoch_CE_loss,
                "contrast_loss": epoch_contrast_loss,
                "epoch": e + 1,
            },
            step=(i + 1) + e * len(train_batch_sampler),
        )
    else:
        wandb.log(
            {
                "train_loss": train_epoch_loss,
                "CE_loss": epoch_CE_loss,
                "epoch": e + 1,
            },
            step=(i + 1) + e * len(train_batch_sampler),
        )

    """
    Validation
    """
    model.eval()
    valid_len = len(valid_batch_sampler)
    eval_loss = torch.tensor([0.0])
    epoch_CE_loss = torch.tensor([0.0])
    epoch_contrast_loss = torch.tensor([0.0])
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader, ncols=100)):
            for key in data.keys():
                if key not in ["name", "indexes", "wer_rank"] and data[key] is not None:

                    data[key] = data[key].to(device)

            if mode == "CONTRAST" or (
                "weightByWER" in train_args.keys()
                and train_args["weightByWER"] == "none"
            ):
                data["wers"] = None

            output = model.forward(**data)

            loss = output["loss"]
            scores = output["score"]

            if mode in ["CONTRAST", "MARGIN"] and output["contrast_loss"] is not None:
                epoch_CE_loss += output["CE_loss"].sum().item()
                epoch_contrast_loss += output["contrast_loss"].sum().item()

            loss = torch.mean(loss)
            if torch.cuda.device_count() > 1:
                loss = loss.sum()
            eval_loss += loss.item()

            """
            Calculate Score
            """
            for n, (name, index, score) in enumerate(
                zip(data["name"], data["indexes"], scores)
            ):
                rescores[index_dict[name]][index] += score.item()

        print(f"Validation: Calcuating CER")
        # best_am = 0.0
        # best_ctc = 0.0
        # best_lm = 0.0
        # best_rescore = 0.0
        # min_cer = 100.0

        best_am, best_ctc, best_lm, best_rescore, min_cer = calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            am_range=[0, 1],
            ctc_range=[0, 1],
            lm_range=[0, 1],
            rescore_range=[0, 1],
            search_step=0.1,
            recog_mode=False,
        )

        print(f"epoch:{e + 1},Validation loss:{eval_loss}")
        print(
            f"epoch:{e + 1},Validation CER:{min_cer}, weight = {[best_am, best_ctc, best_lm, best_rescore]}"
        )

        if mode in ["CONTRAST", "MARGIN"] and output["contrast_loss"] is not None:
            wandb.log(
                {
                    "eval_loss": eval_loss,
                    "eval_CER": min_cer,
                    "CE_loss": epoch_CE_loss,
                    "contrast_loss": epoch_contrast_loss,
                    "epoch": e + 1,
                },
                step=(e + 1) * len(train_batch_sampler),
            )
        else:
            wandb.log(
                {
                    "eval_loss": eval_loss,
                    "CE_loss": eval_loss,
                    "eval_CER": min_cer,
                    "epoch": (e + 1),
                },
                step=((e + 1) * len(train_batch_sampler)),
            )
        logging.warning(f"epoch:{e + 1},validation loss:{eval_loss}")
        logging.warning(
            f"epoch:{e + 1},validation CER:{min_cer} , weight = {[best_am, best_ctc, best_lm, best_rescore]}"
        )

        if (
            mode == "MARGIN"
            and "margin_mode" in train_args.keys()
            and train_args["margin_mode"] in ["WER", "WER_last"]
            and extra_warmup == (train_args["margin_mode"] == "WER")
            and last_val_cer - min_cer < float(train_args["converge"])
        ):
            extra_warmup = not (train_args["margin_mode"])
            print(f"Switch extra_warmup: {train_args['margin_mode']} -> {extra_warmup}")

        last_val_cer = min_cer

        rescores = np.zeros(rescores.shape, dtype=float)

        if save_checkpoint:
            if eval_loss < min_val_loss:
                torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best.pt")
                min_val_loss = eval_loss
            if min_cer < min_val_cer:
                torch.save(
                    checkpoint, f"{checkpoint_path}/checkpoint_train_best_CER.pt"
                )
                min_val_cer = min_cer

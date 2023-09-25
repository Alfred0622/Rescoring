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
from utils.Datasets import prepareSimpleListwiseDataset, prepareListwiseDataset
from utils.CollateFunc import NBestSampler, BatchSampler
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR
from src_utils.get_recog_set import get_valid_set
from utils.PrepareModel import preparePBertSimp
from utils.CollateFunc import SimplePBertBatchWithHardLabel, PBertBatchWithHardLabel
from utils.PrepareScoring import prepare_score_dict, calculate_cer

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

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


config_path = "./config/PBert.yaml"

args, train_args, recog_args = load_config(config_path)

setting = "withLM" if (args["withLM"]) else "noLM"

log_path = f"./log/P_BERT/{args['dataset']}/{setting}/{mode}"
run_name = f"NLP3090_{mode}_batch{train_args['batch_size']}_lr{train_args['lr']}_Freeze{train_args['freeze_epoch']}_{train_args['epoch']}epochs_Reduction{train_args['reduction']}"
if train_args["hard_label"]:
    run_name = run_name + "_HardLabel_Entropy"
else:
    run_name = run_name + train_args["loss_type"]

if (len(sys.argv) >= 3):
    run_name += f"_fromMLMTraining"

if "weightByWER" in train_args.keys() and train_args["weightByWER"] != "none":
    run_name = run_name + f"_weightByWER{train_args['weightByWER']}"
    log_path = log_path + "/weightByWER"
else:
    log_path = log_path + "/normal"

if (train_args['combineScore']):
    run_name = run_name + f"_combineScore"
    log_path = log_path + "/combineScore"
elif (train_args['addLMScore']):
    run_name = run_name + f"_addLMScore"
    log_path = log_path + "_addLMScore"

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
    model, tokenizer = preparePBertSimp(args, train_args, device)
print(type(model))
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
optimizer = AdamW(model.parameters(), lr=float(train_args["lr"]))
# optimizer = Adam(model.parameters(), lr=float(train_args["lr"]))

if "margin_first" in train_args.keys():
    print(f"margina_first:{train_args['margin_first']}")

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
if "WANDB_MODE" in os.environ.keys() and os.environ["WANDB_MODE"] == "disabled":
    get_num = 100
print(f"tokenizing Train")
train_dataset = prepareListwiseDataset(
    data_json=train_json,
    dataset=args["dataset"],
    tokenizer=tokenizer,
    sort_by_len=True,
    topk=int(args['nbest']),
    get_num=get_num,
)
print(f"tokenizing Validation")
valid_dataset = prepareListwiseDataset(
    data_json=valid_json,
    dataset=args["dataset"],
    tokenizer=tokenizer,
    sort_by_len=True,
    topk=int(args['nbest']),
    get_num=get_num,
)

print(f"Prepare Sampler")
train_sampler = NBestSampler(train_dataset)
valid_sampler = NBestSampler(valid_dataset)

print(f"len of sampler:{len(train_sampler)}")

train_batch_sampler = BatchSampler(train_sampler, train_args["batch_size"], batch_by_len=False)
valid_batch_sampler = BatchSampler(valid_sampler, train_args["batch_size"], batch_by_len=False)

print(f"len of batch sampler:{len(train_batch_sampler)}")

train_collate_func = partial(PBertBatchWithHardLabel, use_Margin=False)
valid_collate_func = partial(PBertBatchWithHardLabel, use_Margin=False)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=train_collate_func,
    num_workers=32,
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
    max_lr=float(train_args["lr"]),
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

rescores_flush = rescores.copy()

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

wandb.init(
    project=f"NBestBert_{args['dataset']}_{setting}", config=config, name=run_name
)

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
print(f'run_name:{run_name}')

for e in range(start_epoch, train_args["epoch"]):
    train_epoch_loss = torch.tensor([0.0])
    model.train()

    for i, data in enumerate(tqdm(train_loader, ncols=100)):
        # for key in data.keys():
        #     if key not in ["name", "indexes", "wer_rank"] and data[key] is not None:
        #         # print(f"{key}:{type(data[key])}")
        #         data[key] = data[key].to(device)
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        nBestIndex = data['nBestIndex'].to(device)

        scores = data['asr_score'].to(device)
        am_score = data['am_score'].to(device)
        ctc_score = data['ctc_score'].to(device)
        lm_score = data['lm_score'].to(device)
        labels = data['labels'].to(device)

        # print(f"nBestindex:{data['nBestIndex']}")

        output = model.forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            nBestIndex = nBestIndex,
            am_score = am_score,
            ctc_score = ctc_score,
            lm_score = lm_score,
            scores = scores,
            labels = labels
        )

        loss = output["loss"]
        loss = loss / float(train_args["accumgrad"])

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

    if e == 0 or (e + 1) % 5 == 0:
        checkpoint = {
            "epoch": e,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
        }

        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_{e+1}.pt")

    wandb.log(
        {
            "train_epoch_loss": train_epoch_loss
            / (len(train_batch_sampler) / int(train_args["accumgrad"])),
            "epoch": e + 1,
        },
        step=(i + 1) + e * len(train_batch_sampler),
    )

    """
    Validation
    """
    print(
            f'optimizer:{optimizer.param_groups[0]["lr"]}, scheduler:{lr_scheduler.get_last_lr()}'
        )
    # print(f'loss:{train_epoch_loss}')
    # exit(0)
    model.eval()
    valid_len = len(valid_batch_sampler)
    eval_loss = torch.tensor([0.0])

    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader, ncols=100)):
            # for key in data.keys():
            #     if key not in ["name", "indexes", "wer_rank"] and data[key] is not None:
            #         data[key] = data[key].to(device)
            
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            nBestIndex = data['nBestIndex'].to(device)
            am_score = data['am_score'].to(device)
            ctc_score = data['ctc_score'].to(device)
            lm_score = data['lm_score'].to(device)
            scores = data['asr_score'].to(device)
            labels = data['labels'].to(device)

            output = model.forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
                nBestIndex = nBestIndex,
                am_score = am_score,
                ctc_score = ctc_score,
                lm_score = lm_score,
                scores = scores,
                labels = labels
            )

            loss = output["loss"]
            scores = output["score"]

            # loss = torch.mean(loss)
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

        wandb.log(
            {"eval_CE_loss": eval_loss, "eval_CER": min_cer, "epoch": (e + 1)},
            step=((e + 1) * len(train_batch_sampler)),
        )
        logging.warning(f"epoch:{e + 1},validation loss:{eval_loss}")
        logging.warning(
            f"epoch:{e + 1},validation CER:{min_cer} , weight = {[best_am, best_ctc, best_lm, best_rescore]}"
        )

        last_val_cer = min_cer
        assert(not np.array_equal(rescores, rescores_flush)), "Not Deep Copy"
        rescores = rescores_flush.copy()
        

        if eval_loss < min_val_loss:
            torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best.pt")
            min_val_loss = eval_loss
        if min_cer < min_val_cer:
            torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best_CER.pt")
            min_val_cer = min_cer

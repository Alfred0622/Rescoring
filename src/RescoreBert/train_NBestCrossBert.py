import os
import sys
import torch
import logging
import json
import wandb
import random

sys.path.append("../")
from tqdm import tqdm
from pathlib import Path
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from src_utils.LoadConfig import load_config
from utils.Datasets import prepareListwiseDataset
from utils.CollateFunc import NBestSampler, BatchSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from src_utils.get_recog_set import get_valid_set
from utils.PrepareModel import prepareNBestCrossBert
from utils.CollateFunc import crossNBestBatch
from utils.PrepareScoring import (
    prepare_score_dict,
    calculate_cer,
    calculate_cerOnRank,
    get_result,
)
import math

checkpoint = None

# choose_mode = sys.argv[1].to_lower()
# assert (choose_mode in ['pbert', 'nbestcrossbert']), 'Mode should be PBert or NBestCrossBert'

if len(sys.argv) == 2:
    checkpoint_path = sys.argv[1]

elif len(sys.argv) >= 3:
    assert len(sys.argv) == 3

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

config_path = "./config/NBestCrossBert.yaml"
args, train_args, recog_args = load_config(config_path)
mode = "Normal"

seed = int(args["seed"])
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

print("\n")
for key in train_args.keys():
    print(f"{key}: {train_args[key]}")
print("\n")

# if train_args["useNBestCross"]:
#     if train_args["trainAttendWeight"]:
#         mode = "CrossAttend_TrainWeight"
#     else:
#         mode = "CrossAttend"

#     if train_args["addRes"]:
#         mode = mode + "_ResOnNbest"

#     mode = mode + f"_Att_{train_args['AttLayer']}Layers"

mode = mode + f"_{train_args['fuseType']}"

mode = mode + f"_{train_args['lossType']}"

if train_args["hardLabel"]:
    mode = mode + "_hardLabel"
if train_args["sortByLen"]:
    mode = mode + "_sortByLength"
if train_args["concatCLS"]:
    mode = mode + "_ResCLS"
# if train_args["weightByGrad"]:
#     mode = mode + "_weightByGrad"
if train_args["noCLS"]:
    mode = mode + "_noCLS"
if train_args["noSEP"]:
    mode = mode + "_noSEP"

dropout = 0.1

if "dropout" in train_args.keys():
    dropout = float(train_args["dropout"])
    mode = mode + f"_dropout{train_args['dropout']}"
else:
    mode = mode + f"_dropout0.1"

mode = mode + f"_seed{seed}"

setting = "withLM" if (args["withLM"]) else "noLM"

log_path = Path(f"./log/NBestCrossBert/{args['dataset']}/{setting}/{mode}")
log_path.mkdir(parents=True, exist_ok=True)

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"{log_path}/train_batch{train_args['batch_size']}_{train_args['lr']}.log",
    filemode="w",
    format=FORMAT,
)

valid_set = get_valid_set(args["dataset"])
test_set = ["test"]

"""
model and Optimizer initilaization
"""
model, tokenizer = prepareNBestCrossBert(
    args["dataset"],
    device,
    lstm_dim=train_args["lstm_embedding"],
    fuseType=train_args["fuseType"],
    lossType="Entropy" if train_args["hardLabel"] else train_args["lossType"],
    concatCLS=train_args["concatCLS"],
    dropout=dropout,
    noCLS=train_args["noCLS"],
    noSEP=train_args["noSEP"],
)

model = model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
if train_args["fuseType"] == "lstm" and len(sys.argv) == 2:
    checkpoint = torch.load(checkpoint_path)
    print(f"load LSTM")
    # print(checkpoint['model'].keys())
    model.lstm.load_state_dict(checkpoint["model"]["LSTM"])
    model.concatLinear.load_state_dict(checkpoint["model"]["concatLSTM"])


optimizer = AdamW(model.parameters(), lr=float(train_args["lr"]))
# optimizer = Adam(model.parameters(), lr=float(train_args["lr"]))

with open(f"../../data/{args['dataset']}/data/{setting}/train/data.json") as f, open(
    f"../../data/{args['dataset']}/data/{setting}/{valid_set}/data.json"
) as dev :
    train_json = json.load(f)
    valid_json = json.load(dev)

start_epoch = 0
get_num = -1

"""
Load checkpoint
"""

if "WANDB_MODE" in os.environ.keys():
    if os.environ["WANDB_MODE"] == "disabled":
        get_num = 100

print(f"tokenizing Train")
train_dataset = prepareListwiseDataset(
    data_json=train_json,
    tokenizer=tokenizer,
    dataset=args["dataset"],
    topk = int(args['nbest']),
    sort_by_len=train_args["sortByLen"],
    get_num=get_num,
    maskEmbedding=train_args["fuseType"] == "query",
)
print(f"tokenizing Validation")
valid_dataset = prepareListwiseDataset(
    data_json=valid_json,
    dataset=args["dataset"],
    tokenizer=tokenizer,
    topk = int(args['nbest']),
    sort_by_len=train_args["sortByLen"],
    get_num=get_num,
    maskEmbedding=train_args["fuseType"] == "query",
)

print(f"Prepare Sampler")
train_sampler = NBestSampler(train_dataset)
valid_sampler = NBestSampler(valid_dataset)


print(f"len of sampler:{len(train_sampler)}")

train_batch_sampler = BatchSampler(
    train_sampler,
    train_args["batch_size"],
    batch_by_len=(train_args["fuseType"] in ["attn", "lstm"]),
)
valid_batch_sampler = BatchSampler(
    valid_sampler,
    train_args["valid_batch"],
    batch_by_len=(train_args["fuseType"] in ["attn", "lstm"]),
)

print(f"len of batch sampler:{len(train_batch_sampler)}")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=partial(crossNBestBatch, hard_label=train_args["hardLabel"]),
    num_workers=16,
    pin_memory=True,
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_sampler=valid_batch_sampler,
    collate_fn=partial(crossNBestBatch, hard_label=train_args["hardLabel"]),
    num_workers=16,
    pin_memory=True,
)

if (args['dataset'] not in ['librispeech', 'aishell2', 'csj']):
    with open(f"../../data/{args['dataset']}/data/{setting}/test/data.json") as test:
        test_json = json.load(test)
        test_dataset = prepareListwiseDataset(
            data_json=test_json,
            dataset=args["dataset"],
            tokenizer=tokenizer,
            topk = int(args['nbest']),
            sort_by_len=train_args["sortByLen"],
            get_num=get_num,
            maskEmbedding=train_args["fuseType"] == "query",
        )
        test_sampler = NBestSampler(test_dataset)
        test_batch_sampler = BatchSampler(
            test_sampler,
            train_args["valid_batch"],
            batch_by_len=(train_args["fuseType"] in ["attn", "lstm"]),
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_sampler=test_batch_sampler,
            collate_fn=partial(crossNBestBatch, hard_label=train_args["hardLabel"]),
            num_workers=16,
            pin_memory=True,
        )
        (
            test_index_dict,
            test_inverse_dict,
            test_am_scores,
            test_ctc_scores,
            test_lm_scores,
            test_rescores,
            test_wers,
            test_hyps,
            test_refs,
        ) = prepare_score_dict(test_json, nbest=args["nbest"])
        test_rescores_flush = test_rescores.copy()

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
config = dict()
# if train_args["useNBestCross"]:
#     config = {
#         "args": args,
#         "train_args": train_args,
#         "Bert_config": model.bert.config.to_dict()
#         if (torch.cuda.device_count() <= 1)
#         else model.module.bert.config.to_dict(),
#         "CrossAttentionConfig": model.fuseAttention.config
#         if (torch.cuda.device_count() <= 1)
#         else model.module.fuseAttention.config,
#     }
# else:
config = {
    "args": args,
    "train_args": train_args,
    "Bert_config": model.bert.config.to_dict()
    if (torch.cuda.device_count() <= 1)
    else model.module.bert.config.to_dict(),
}

wandb.init(
    project=f"myBert_{args['dataset']}_{setting}",
    config=config,
    name=f"{mode}_batch{train_args['batch_size']}_lr{train_args['lr']}_freeze{train_args['freeze_epoch']}_warmup{train_args['warmup_ratio']}",
)

checkpoint_path = Path(
    f"./checkpoint/{args['dataset']}/NBestCrossBert/{setting}/{mode}/{args['nbest']}best/batch{train_args['batch_size']}_lr{train_args['lr']}_warmup{train_args['warmup_ratio']}_freeze{train_args['freeze_epoch']}"
)
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
cls_logging = torch.tensor([0.0])
mask_logging = torch.tensor([0.0])

for param in model.bert.parameters():
    param.requires_grad = False

if train_args["fuseType"] == "query" and not train_args["concatCLS"]:
    if train_args["train_cls_first"]:
        get_cls_loss = True
        get_mask_loss = False
    else:
        get_cls_loss = True
        get_mask_loss = True
else:
    get_cls_loss = False
    get_mask_loss = False

for e in range(start_epoch, train_args["epoch"]):
    model.train()
    epoch_loss = torch.tensor([0.0])
    cls_loss = 0.0
    mask_loss = 0.0

    if e >= int(train_args["freeze_epoch"]):
        print(f"unfreeze bert")
        for param in model.bert.parameters():
            param.requires_grad = True
    else:
        print(f"freeze bert")

    if train_args["fuseType"] == "query" and train_args["train_cls_first"]:
        if e > 10:
            get_cls_loss = False
            get_mask_loss = True

    for i, data in enumerate(tqdm(train_loader, ncols=100)):
        for key in data.keys():
            if key not in ["name", "indexes"]:
                # print(f"{key}:{type(data[key])}")
                data[key] = data[key].to(device)

        output = model.forward(
            **data,
            use_cls_loss=get_cls_loss,
            use_mask_loss=get_mask_loss,
        )

        loss = output["loss"]
        loss = torch.mean(loss)

        if torch.cuda.device_count() > 1:
            loss = loss.sum()
        loss.backward()

        if ((i + 1) % int(train_args["accumgrad"])) == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

        if (step > 0) and step % int(train_args["print_loss"]) == 0:
            logging_loss = logging_loss / step
            cls_logging = cls_logging / step
            mask_logging = mask_logging / step
            logging.warning(f"epoch:{e + 1} step {i + 1},loss:{logging_loss}")

            if "cls_loss" in output.keys():
                wandb.log(
                    {
                        "train_loss": logging_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "cls_loss": cls_logging,
                        "mask_loss": mask_logging,
                    },
                    step=(i + 1) + e * len(train_batch_sampler),
                )
            else:
                wandb.log(
                    {"train_loss": logging_loss, "lr": optimizer.param_groups[0]["lr"]},
                    step=(i + 1) + e * len(train_batch_sampler),
                )

            logging_loss = torch.tensor([0.0])
            cls_logging = torch.tensor([0.0])
            mask_logging = torch.tensor([0.0])
            step = 0

        logging_loss += loss.item()
        epoch_loss += loss.item()

        if "cls_loss" in output.keys():
            cls_logging += output["cls_loss"].item()
            mask_logging += output["mask_loss"].item()
            cls_loss += output["cls_loss"].item()
            mask_loss += output["mask_loss"].item()

    checkpoint = {
        "epoch": e,
        "withLM": "withLM" if (args["withLM"]) else "noLM",
        "train_args": train_args,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
    }

    if (e + 1) % 5 == 0 or e == 0:
        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_{e+1}.pt")
    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_last.pt")

    logging.warning(
        f"epoch {e + 1}, training loss:{epoch_loss / len(train_batch_sampler)}"
    )
    if "cls_loss" in output.keys():
        wandb.log(
            {
                "train_loss_per_epoch": (epoch_loss / len(train_batch_sampler)),
                "cls_loss_per_epoch": (cls_loss / len(train_batch_sampler)),
                "mask_loss_per_epoch": (mask_loss / len(train_batch_sampler)),
                "epoch": (e + 1),
            }
        )

    else:
        wandb.log(
            {
                "cls_loss_per_epoch": (epoch_loss / len(train_batch_sampler)),
                "epoch": (e + 1),
            }
        )
    """
    Validation
    """
    model.eval()
    valid_len = len(valid_batch_sampler)
    eval_loss = torch.tensor([0.0])
    cls_loss = torch.tensor([0.0])
    mask_loss = torch.tensor([0.0])

    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader, ncols=100)):
            for key in data.keys():
                if key not in ["name", "indexes"]:
                    data[key] = data[key].to(device)

            output = model.forward(
                **data,
                use_cls_loss=get_cls_loss,
                use_mask_loss=get_mask_loss,
            )

            loss = output["loss"]
            scores = output["score"]
            loss = torch.mean(loss)
            if torch.cuda.device_count() > 1:
                loss = loss.sum()
            eval_loss += loss.item()

            if "cls_loss" in output.keys():
                cls_loss += output["cls_loss"].item()
                mask_loss += output["mask_loss"].item()

            for n, (name, index, score) in enumerate(
                zip(data["name"], data["indexes"], scores)
            ):
                rescores[index_dict[name]][index] += score.item()
        if "cls_loss" in output.keys() and train_args["weightByGrad"]:
            model.set_weight(cls_loss, mask_loss)

        """
        Calculate Score
        """

        print(f"Validation: Calcuating CER")
        # if not train_args["sepTask"] or not train_args["scoreByRank"]:
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
            search_step=0.2,
            recog_mode=False,
        )
        print(
            f"epoch:{e + 1},Validation CER:{min_cer}, weight = {[best_am, best_ctc, best_lm, best_rescore]}"
        )


        # else:
        #     min_cer = calculate_cerOnRank(
        #         am_scores, ctc_scores, lm_scores, rescores, wers, withLM=args["withLM"]
        # )
        print(f"epoch:{e + 1},Validation CER:{min_cer}")
        print(f"epoch:{e + 1},Validation loss:{eval_loss}")

        if (args['dataset'] not in ['librispeech', 'aishell2', 'csj']):
            print(f"Test by epoch")
            for i, data in enumerate(tqdm(test_loader, ncols=100)):
                for key in data.keys():
                    if key not in ["name", "indexes"]:
                        data[key] = data[key].to(device)
                output = model.forward(
                    **data,
                    use_cls_loss=get_cls_loss,
                    use_mask_loss=get_mask_loss,
                )

                scores = output["score"]
                for n, (name, index, score) in enumerate(
                    zip(data["name"], data["indexes"], scores)
                ):
                    test_rescores[test_index_dict[name]][index] += score.item()
            test_cer, _ = get_result(
                test_index_dict,
                test_am_scores,
                test_ctc_scores,
                test_lm_scores,
                test_rescores,
                test_wers,
                test_hyps,
                test_refs,
                best_am,
                best_ctc,
                best_lm,
                best_rescore,
            )

        if "cls_loss" in output.keys():
            wandb.log(
                {
                    "eval_loss": eval_loss / len(valid_batch_sampler),
                    "eval_CER": min_cer,
                    "test_CER": test_cer,
                    "epoch": (e + 1),
                    "eval_cls_loss": cls_loss / len(valid_batch_sampler),
                    "eval_mask_loss": mask_loss / len(valid_batch_sampler),
                    "clsWeight": model.clsWeight
                    if (hasattr(model, "clsWeight"))
                    else None,
                    "maskWeight": model.maskWeight
                    if (hasattr(model, "maskWeight"))
                    else None,
                },
                step=((e + 1) * len(train_batch_sampler)),
            )

        else:
            wandb.log(
                {
                    "eval_loss": eval_loss / len(valid_batch_sampler),
                    "eval_CER": min_cer,
                    "epoch": (e + 1),
                },
                step=((e + 1) * len(train_batch_sampler)),
            )
        logging.warning(f"epoch:{e + 1},validation loss:{eval_loss}")

        rescores = rescores_flush.copy()
        # test_rescores = test_rescores.copy()

        if eval_loss < min_val_loss:
            torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best.pt")
            min_val_loss = eval_loss
        if min_cer < min_val_cer:
            torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best_CER.pt")
            min_val_cer = min_cer

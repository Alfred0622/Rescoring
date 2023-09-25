import os
import argparse
import json
import sys

sys.path.append("..")
import torch
from tqdm import tqdm
import logging
import torch
import random
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from utils.Datasets import getRescoreDataset
from src_utils.LoadConfig import load_config
from utils.CollateFunc import (
    MDTrainBatch,
    RescoreBertBatch,
    NBestSampler,
    RescoreBert_BatchSampler,
    MWERBatch,
)
from utils.PrepareModel import prepare_RescoreBert
from utils.DataPara import BalancedDataParallel
from utils.PrepareScoring import prepareRescoreDict, prepare_score_dict, calculate_cer
from torch.optim import AdamW
import gc
from jiwer import wer, cer
import wandb

mode = sys.argv[1]
checkpoint = None
if len(sys.argv) == 3:
    print(f"load_checkpoint")
    checkpoint = sys.argv[2]

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if len(sys.argv) != 2 and len(sys.argv) != 3:
    assert (
        len(sys.argv) == 2
    ), "python ./train_RescoreBert.py {MD,MWER,MWED} checkpoint_Path(Optional)"

print(f"mode:{mode}")
use_MWER = False
use_MWED = False
if mode == "MWER":
    use_MWER = True
elif mode == "MWED":
    use_MWED = True
else:
    mode = "MD"

config = f"./config/RescoreBert.yaml"
args, train_args, recog_args = load_config(config)

setting = "withLM" if (args["withLM"]) else "noLM"

log_path = Path(f"./log/RescoreBert/{args['dataset']}/{setting}/{mode}")
log_path.mkdir(parents=True, exist_ok=True)

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"{log_path}/train_batch{train_args['train_batch']}_accum{train_args['accumgrad']}Grads_{train_args['lr']}_reduction{train_args['reduction']}",
    filemode="w",
    format=FORMAT,
)

if args["dataset"] in ["aishell2"]:
    dev_set = "dev_ios"
elif args["dataset"] in ["librispeech"]:
    dev_set = "valid"
else:
    dev_set = "dev"

model, tokenizer = prepare_RescoreBert(args["dataset"], device, train_args['reduction'])
model = model.to(device)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
optimizer = AdamW(model.parameters(), lr=float(train_args["lr"]))

start_epoch = 0
if checkpoint is not None:
    checkpoint = torch.load(checkpoint)
    print(checkpoint.keys())
    start_epoch = checkpoint["epoch"]
    if torch.cuda.device_count() > 1:
        model.module.bert.load_state_dict(checkpoint["bert"])
        model.module.linear.load_state_dict(checkpoint["fc"])
    else:
        model.bert.load_state_dict(checkpoint["bert"])
        model.linear.load_state_dict(checkpoint["fc"])
    optimizer.load_state_dict(checkpoint["optimizer"])

if not os.path.exists(f"./log/RescoreBert/{args['dataset']}/{mode}/{setting}"):
    os.makedirs(f"./log/RescoreBert/{args['dataset']}/{mode}/{setting}")

"""
    Build the score dictionary
"""

with open(
    f"./data/{args['dataset']}/{setting}/{int(args['nbest'])}best/MLM/train/rescore_data.json"
) as f, open(
    f"./data/{args['dataset']}/{setting}/{int(args['nbest'])}best/MLM/{dev_set}/rescore_data.json"
) as dev:
    train_json = json.load(f)
    valid_json = json.load(dev)

if "WANDB_MODE" in os.environ.keys() and os.environ["WANDB_MODE"] == "disabled":
    fetch_num = 100
else:
    fetch_num = -1

print(f'topk:{args["nbest"]}')

print(f" Tokenization : train")
train_dataset = getRescoreDataset(
    train_json, args["dataset"], tokenizer, topk=args["nbest"], mode=mode, fetch_num = fetch_num
)
print(f" Tokenization : valid")
valid_dataset = getRescoreDataset(
    valid_json, args["dataset"], tokenizer, topk=args["nbest"], mode=mode, fetch_num = fetch_num
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
(
    valid_name_index,
    valid_name_inverse,
    hyps_dict,
    valid_score,
    valid_rescore,
    eval_wers,
) = prepareRescoreDict(valid_json)

if use_MWED or use_MWER:
    train_sampler = NBestSampler(train_dataset)
    valid_sampler = NBestSampler(valid_dataset)

    train_batch_sampler = RescoreBert_BatchSampler(
        train_sampler, train_args["train_batch"]
    )
    valid_batch_sampler = RescoreBert_BatchSampler(valid_sampler, 1)

    train_loader = DataLoader(
        dataset=train_dataset,
        # batch_size=train_args['train_batch'],
        batch_sampler=train_batch_sampler,
        collate_fn=MWERBatch,
        num_workers=16,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        # batch_size = 1,
        batch_sampler=valid_batch_sampler,
        collate_fn=MWERBatch,
        num_workers=16,
        pin_memory=True,
    )

    print(f"sampler:{len(valid_sampler)}")
    print(f"RescoreBert_BatchSampler:{len(valid_batch_sampler)}")
    print(f"Loader:{len(valid_loader)}")
    # exit(0)

else:
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_args["train_batch"],
        # sampler = train_sampler,
        collate_fn=MDTrainBatch,
        num_workers=16,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=train_args["train_batch"],
        # sampler = valid_sampler,
        collate_fn=MDTrainBatch,
        num_workers=16,
        pin_memory=True,
    )

weight = 1e-4
score_weight = 1.0
weight = torch.tensor(weight, dtype=torch.float32).to(device)

wandb_config = wandb.config
wandb_config = (
    model.bert.config if (torch.cuda.device_count() == 1) else model.module.bert.config
)
wandb_config.learning_rate = float(train_args["lr"])
wandb_config.batch_size = train_args["train_batch"]
cal_batch = int(train_args["train_batch"]) * train_args["accumgrad"]

wandb.init(
    project=f"Rescore_{args['dataset']}_{setting}",
    config=wandb_config,
    name=f"{args['nbest']}Best_RescoreBert_{mode}_batch{train_args['train_batch']}_accum{train_args['accumgrad']}grads_lr{train_args['lr']}_reduction{train_args['reduction']}",
)

optimizer.zero_grad(set_to_none=True)

checkpoint_path = Path(
    f"./checkpoint/{args['dataset']}/RescoreBert/{setting}/{mode}/{args['nbest']}best/batch{train_args['train_batch']}_accum{train_args['accumgrad']}grads_lr{train_args['lr']}_reduction{train_args['reduction']}"
)
checkpoint_path.mkdir(parents=True, exist_ok=True)

wandb.watch(model, log_freq=int(train_args["print_loss"]))

step = 0
min_val_loss = 1e8
min_val_cer = 1e6
logging_loss = torch.tensor([0.0], device=device)
for e in range(start_epoch, train_args["epoch"]):
    model.train()

    print(f"score_weight:{score_weight}")

    for i, data in enumerate(tqdm(train_loader, ncols=100)):
        # print(f"name:{data['name']}")
        data["input_ids"] = data["input_ids"].to(device)
        data["attention_mask"] = data["attention_mask"].to(device)
        data["labels"] = data["labels"].to(device)

        output = model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"],
        )

        loss = output["loss"] / int(train_args["accumgrad"])
        loss = loss.sum()

        # MWER
        if mode == "MWER":
            data["wer"] = data["wer"].to(device)
            first_score = data["score"].to(device)

            combined_score = first_score + score_weight * output["score"].clone()

            avg_error = data["avg_error"].to(device)

            # softmax seperately
            index = 0
            for nbest in data["nbest"]:

                combined_score[index : index + nbest] = torch.softmax(
                    combined_score[index : index + nbest].clone(), dim=-1
                )

                index = index + nbest

            loss_MWER = combined_score * (data["wer"] - avg_error)

            loss_MWER = loss_MWER.sum()

            loss = loss * 1e-4
            loss = loss_MWER + loss

        elif mode == "MWED":
            data["wer"] = data["wer"].to(device, non_blocking=True)
            first_score = data["score"].to(device)
            wer = data["wer"].clone()

            assert (
                first_score.shape == output["score"].shape
            ), f"first_score:{first_score.shape}, score:{output['score'].shape}"

            combined_score = first_score + score_weight * output["score"].clone()

            assert (
                combined_score.shape == data["wer"].shape
            ), f"combined_score:{combined_score.shape}, wer:{data['wer'].shape}"
            # calculate Temperature T
            index = 0
            scoreSum = torch.tensor([]).to(device)
            werSum = torch.tensor([]).to(device)
            for nbest in data["nbest"]:
                score_sum = torch.sum(
                    combined_score[index : index + nbest].clone()
                ).repeat(nbest)

                wer_sum = torch.sum(wer[index : index + nbest].clone()).repeat(nbest)

                scoreSum = torch.cat([scoreSum, score_sum])
                werSum = torch.cat([werSum, wer_sum])

                index = index + nbest

            assert (
                scoreSum.shape == combined_score.shape
            ), f"combined_score:{combined_score.shape}, scoreSum:{scoreSum.shape}"

            index = 0
            T = scoreSum / werSum  # Temperature T

            # print(f"T:{T}")

            combined_score = combined_score / T
            assert (
                scoreSum.shape == combined_score.shape
            ), f"scoreSum:{scoreSum.shape} != combined_score:{combined_score}"
            assert (
                werSum.shape == combined_score.shape
            ), f"werSum:{werSum.shape} != combined_score:{combined_score}"
        
            # print(f"combined_score:{combined_score}")
            # print(f"wer:{wer}")

            combined_score_before = combined_score.clone()
            wer_before = wer.clone()

            for nbest in data["nbest"]:
                combined_score[index : index + nbest] = torch.softmax(
                    combined_score[index : index + nbest].clone(), dim=-1
                )
                wer[index : index + nbest] = torch.softmax(
                    wer[index : index + nbest].clone(), dim=-1
                )

                index = index + nbest
            
            # print(f"combined_score after softmax:{combined_score}")
            # print(f"wer after softmax:{wer}")

            loss_MWED = wer * torch.log(combined_score)
            loss_MWED = loss_MWED.sum()
            loss_MWED = torch.neg(loss_MWED)

            assert not (
                torch.isnan(loss) or torch.isnan(loss_MWED)
            ), f"name:{data['name']}, \nloss:{loss}, \nloss_MWED:{loss_MWED}, \nT:{T}, score:{combined_score}, wer:{wer}\n combined_score before softmax:{combined_score_before}\n wers before softmax:{wer_before}"

            loss = loss_MWED + 1e-4 * loss

        if torch.cuda.device_count() > 1:
            loss = loss.sum()

        loss = loss / float(train_args["accumgrad"])
        loss.backward()

        if ((i + 1) % int(train_args["accumgrad"])) == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

        if step > 0 and (step % int(train_args["print_loss"]) == 0):
            logging.warning(f"step {i + 1},loss:{logging_loss / step}")
            wandb.log(
                {
                    "train_loss": (logging_loss / step),
                },
                step=(i + 1) + e * len(train_loader),
            )
            logging_loss = torch.tensor([0.0], device=device)
            step = 0

        logging_loss += loss.clone().detach()

    checkpoint = {
        "epoch": e,
        "bert": model.bert.state_dict()
        if torch.cuda.device_count() == 1
        else model.module.bert.state_dict(),
        "fc": model.linear.state_dict()
        if torch.cuda.device_count() == 1
        else model.module.linear.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_{e+1}.pt")

    print(f"epoch:{e + 1} validation")
    with torch.no_grad():
        eval_loss = torch.tensor([0.0], device=device)
        model.eval()
        for i, data in enumerate(tqdm(valid_loader, ncols=100)):
            data["input_ids"] = data["input_ids"].to(device)
            data["attention_mask"] = data["attention_mask"].to(device)
            data["labels"] = data["labels"].to(device)

            output = model(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                labels=data["labels"],
            )
            loss = output["loss"]

            if mode in ["MWER", "MWED"]:
                for n, (name, index,score) in enumerate(zip(data["name"],data['index'], output["score"])):
                    # print(F'index:{index}')
                    # print(f"rescores:{rescores[index_dict[name]][index]}")
                    rescores[index_dict[name]][index] = score.item()
                    valid_rescore[valid_name_index[name]][index] = score.item()
                    # print(f"rescores after:{rescores[index_dict[name]]}")

            else:
                # print(f"name:{len(data['name'])}")
                # print(f"score:{len(output['score'])}")
                for n, (name, index, score) in enumerate(zip(data["name"], data['index'], output["score"])):
                    rescores[index_dict[name]][index] = score.item()

            if mode == "MWER":
                data["wer"] = data["wer"].to(device)
                first_score = data["score"].to(device)

                combined_score = first_score + score_weight * output["score"].clone()

                avg_error = data["avg_error"].to(device)

                # softmax seperately
                index = 0
                for nbest in data["nbest"]:

                    combined_score[index : index + nbest] = torch.softmax(
                        combined_score[index : index + nbest], dim=-1
                    )

                    index = index + nbest

                loss_MWER = first_score * (data["wer"] - avg_error)
                loss_MWER = torch.sum(loss_MWER)

                # print(f'loss_MWER:{loss_MWER}')

                loss = loss_MWER + 1e-4 * loss

            elif mode == "MWED":
                data["wer"] = data["wer"].to(device)
                first_score = data["score"].to(device)
                wer = data["wer"].clone()

                assert (
                    first_score.shape == output["score"].shape
                ), f"first_score:{first_score.shape}, score:{output['score'].shape}"

                combined_score = first_score + score_weight * output["score"].clone()

                index = 0
                scoreSum = torch.tensor([]).to(device)
                werSum = torch.tensor([]).to(device)

                for n, (name, score) in enumerate(zip(data["name"], output["score"])):
                    valid_rescore[valid_name_index[name]][n] = score.item()

                for nbest in data["nbest"]:

                    score_sum = torch.sum(
                        combined_score[index : index + nbest].clone()
                    ).repeat(nbest)

                    wer_sum = torch.sum(wer[index : index + nbest].clone()).repeat(
                        nbest
                    )

                    scoreSum = torch.cat([scoreSum, score_sum])
                    werSum = torch.cat([werSum, wer_sum])

                    index = index + nbest

                index = 0

                T = scoreSum / werSum  # hyperparameter T

                combined_score = combined_score / T

                for nbest in data["nbest"]:
                    combined_score[index : index + nbest] = torch.softmax(
                        combined_score[index : index + nbest].clone(), dim=-1
                    )
                    wer[index : index + nbest] = torch.softmax(
                        wer[index : index + nbest].clone(), dim=-1
                    )

                    index = index + nbest

                # print(f'combined_score after scale & softmax:{combined_score}')
                loss_MWED = wer * torch.log(combined_score)
                loss_MWED = torch.neg(loss_MWED)
                loss_MWED = torch.sum(loss_MWED)
                loss = loss_MWED + 1e-4 * loss

            # print(f'total_loss:{loss}')

            if torch.cuda.device_count() > 1:
                loss = loss.sum()
            eval_loss += loss.item()

        if mode in ["MWER", "MWED"]:
            min_cer = 100
            best_weight = score_weight
            for w in np.arange(0, 1.1, step=0.1):
                c = 0
                s = 0
                d = 0
                i = 0
                for n, (score, rescore) in enumerate(zip(valid_score, valid_rescore)):

                    combined_score = score + w * rescore
                    combined_score[np.isnan(combined_score)] = np.NINF
                    best_index = np.argmax(combined_score)

                    best_wers = eval_wers[n][best_index]

                    c += best_wers["hit"]
                    s += best_wers["sub"]
                    d += best_wers["del"]
                    i += best_wers["ins"]

                cer = (s + d + i) / (c + s + d)
                if cer < min_cer:
                    best_weight = w
                    min_cer = cer

            score_weight = best_weight
            print(f"epoch: {e + 1}, Best Weight:{best_weight} , min_CER:{min_cer}")

        best_am, best_ctc, best_lm, best_rescore, eval_cer = calculate_cer(
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

        eval_loss = (
            (eval_loss / len(valid_loader))
            if (mode == "MD")
            else (eval_loss / len(valid_batch_sampler))
        )
        print(f"epoch:{e + 1}, loss:{eval_loss}")
        wandb.log(
            {"eval_loss": eval_loss, "eval_cer": eval_cer, "epoch": (e + 1)},
            step=(e + 1) * len(train_loader),
        )
        logging.warning(f"epoch:{e + 1}, loss:{eval_loss}")

        if eval_loss < min_val_loss:
            torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best.pt")
            min_val_loss = eval_loss
        if eval_cer < min_val_cer:
            torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best_CER.pt")
            min_val_cer = eval_cer

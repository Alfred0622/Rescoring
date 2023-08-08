import json
import sys

sys.path.append("../")
import torch
import numpy as np
from numba import jit, njit
from torch.utils.data import DataLoader
import os

from src_utils.LoadConfig import load_config
from utils.Datasets import prepareListwiseDataset
from utils.CollateFunc import NBestSampler, BatchSampler, PBertBatchWithHardLabel
from utils.PrepareModel import preparePBertSimp
from utils.PrepareScoring import calculate_cer, get_result, prepare_score_dict
from src_utils.get_recog_set import get_recog_set
from pathlib import Path
from tqdm import tqdm
from functools import partial
import time

config_path = "./config/PBert.yaml"
args, train_args, recog_args = load_config(config_path)
mode = ""
if train_args["hard_label"]:
    mode = "_HardLabel"
else:
    mode = ""

checkpoint_path = sys.argv[1]
decode_mode = sys.argv[2].strip().upper()
assert decode_mode in [
    "PBERT",
    "PBERT_ENTROPY",
], "decode mode only has PBERT, PBERT_ENTROPY"
mode = "PBERT" + mode

setting = "withLM" if (args["withLM"]) else "noLM"
print(f"{args['dataset']} : {setting}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = preparePBertSimp(args, train_args, device = device)
model = model.to(device)
model.eval()
checkpoint = torch.load(checkpoint_path)
print(f"checkpoint:{checkpoint.keys()}")
print(f"checkpoint[model]: {checkpoint['model'].keys()}")
model.load_state_dict(checkpoint["model"])

recog_set = get_recog_set(args["dataset"])
dev_set = recog_set[0]

for_train = True
if (for_train):
    recog_set = ['train']

best_am = 0.0
best_ctc = 0.0
best_lm = 0.0
best_rescore = 0.0

for task in recog_set:
    total_time = 0.0
    recog_path = f"../../data/{args['dataset']}/data/{setting}/{task}/data.json"
    with open(recog_path) as f:
        recog_json = json.load(f)

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
        ) = prepare_score_dict(recog_json, nbest=args["nbest"])

        data_num = 0

        recog_dataset = prepareListwiseDataset(
            recog_json, args["dataset"], tokenizer, sort_by_len=True, topk = int(args['nbest']),get_num=-1
        )
        recog_sampler = NBestSampler(recog_dataset)
        recog_batch_sampler = BatchSampler(recog_sampler, recog_args["batch"])
        recog_loader = DataLoader(
            dataset=recog_dataset,
            batch_sampler=recog_batch_sampler,
            collate_fn=partial(PBertBatchWithHardLabel, use_Margin = False),
            num_workers=16,
        )
        name_set = set()
        with torch.no_grad():
            for data in tqdm(recog_loader, ncols=100):
                # for key in data.keys():
                #     if key not in ["name", "indexes"] and data[key] is not None:
                #         data[key] = data[key].to(device)
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                nBestIndex = data['nBestIndex'].to(device)

                scores = data['asr_score'].to(device)
                am_score = data['am_score'].to(device)
                ctc_score = data['ctc_score'].to(device)
                data_num += 1

                if decode_mode == "PBERT_ENTROPY":
                    output = model.recognize_by_attention(
                        input_ids=data["input_ids"],
                        attention_mask=data["attention_mask"],
                    )["score"]
                else:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    output = model.recognize(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        am_score=am_score,
                        ctc_score=ctc_score,
                        scores = scores,
                    )["score"]
                    torch.cuda.synchronize()
                    t1 = time.time()
                    run_time = t1 - t0
                    total_time += run_time

                # print(f"output:{output.shape}\n {output}")
                # print(f"name : {data['name']}")
                # print(f"index : {data['indexes']}")

                for n, (name, index, score) in enumerate(
                    zip(data["name"], data["indexes"], output)
                ):
                    assert torch.isnan(score).count_nonzero() == 0, "got nan"
                    rescores[index_dict[name]][index] += score.item()
                    name_set.add(name)
                # print(f"rescores: {rescores[index_dict[name]]}")

        # print(f"name : {data['name']}")
        # print(f"index : {data['indexes']}")
        # print(f"score:{output}")
        save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}/{args['nbest']}/PBERT")
        save_path.mkdir(exist_ok=True, parents=True)

        rescore_data = []
        for name in name_set:
            rescore_data.append(
                {
                    "name": name,
                    "hyps": hyps[index_dict[name]],
                    "ref": hyps[index_dict[name]],
                    "rescore": rescores[index_dict[name]].tolist()
                }
            )
        save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}/{args['nbest']}best/PBERT")
        save_path.mkdir(exist_ok=True, parents=True)

        with open(f"{save_path}/data.json", "w") as f:
            json.dump(rescore_data, f, ensure_ascii=False, indent=1)

        if task == dev_set:
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
            )

            print(
                f"am_weight:{best_am}\n ctc_weight:{best_ctc}\n lm_weight:{best_lm}\n rescore_weight:{best_rescore}\n CER:{min_cer}"
            )
        if (not for_train):
            cer, result_dict = get_result(
                inverse_dict,
                am_scores,
                ctc_scores,
                lm_scores,
                rescores,
                wers,
                hyps,
                refs,
                am_weight=best_am,
                ctc_weight=best_ctc,
                lm_weight=best_lm,
                rescore_weight=best_rescore,
            )

            print(f"Dataset:{args['dataset']}")
            print(f"setting:{setting}")
            print(f"task:{task}")
            print(f"CER : {cer}")

            save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}/{args['nbest']}best/PBERT")
            save_path.mkdir(exist_ok=True, parents=True)
            with open(f"{save_path}/analysis.json", "w") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=1)

        avg_time=total_time / data_num
        print(f"average decode time:{avg_time}")
        


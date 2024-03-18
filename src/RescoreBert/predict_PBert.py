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
from utils.CollateFunc import NBestSampler, BatchSampler, PBertBatch
from utils.PrepareModel import preparePBert
from utils.PrepareScoring import calculate_cer, get_result, prepare_score_dict
from src_utils.get_recog_set import get_recog_set
from pathlib import Path
from tqdm import tqdm

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
model, tokenizer = preparePBert(args, train_args, device = device)
model = model.to(device)
model.eval()
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
print(f"checkpoint:{checkpoint.keys()}")
print(f"checkpoint[model]: {checkpoint['model'].keys()}")
model.load_state_dict(checkpoint["model"])

model.show_param()
exit()


recog_set = get_recog_set(args["dataset"])
dev_set = recog_set[0]

best_am = 0.0
best_ctc = 0.0
best_lm = 0.0
best_rescore = 0.0

for task in recog_set:
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

        recog_dataset = prepareListwiseDataset(
            recog_json, args["dataset"], tokenizer, sort_by_len=True, get_num=-1
        )
        recog_sampler = NBestSampler(recog_dataset)
        recog_batch_sampler = BatchSampler(recog_sampler, recog_args["batch"])
        recog_loader = DataLoader(
            dataset=recog_dataset,
            batch_sampler=recog_batch_sampler,
            collate_fn=PBertBatch,
            num_workers=16,
        )
        with torch.no_grad():
            for data in tqdm(recog_loader, ncols=100):
                for key in data.keys():
                    if key not in ["name", "indexes"] and data[key] is not None:
                        data[key] = data[key].to(device)

                if decode_mode == "PBERT_ENTROPY":
                    output = model.recognize_by_attention(
                        input_ids=data["input_ids"],
                        attention_mask=data["attention_mask"],
                    )["score"]
                else:
                    output = model(
                        input_ids=data["input_ids"],
                        attention_mask=data["attention_mask"],
                        am_score=data["am_score"],
                        ctc_score=data["ctc_score"],
                        nBestIndex=None,
                    )["score"]

                # print(f"output:{output.shape}\n {output}")
                # print(f"name : {data['name']}")
                # print(f"index : {data['indexes']}")

                for n, (name, index, score) in enumerate(
                    zip(data["name"], data["indexes"], output)
                ):
                    assert torch.isnan(score).count_nonzero() == 0, "got nan"
                    rescores[index_dict[name]][index] += score.item()
                # print(f"rescores: {rescores[index_dict[name]]}")

        # print(f"name : {data['name']}")
        # print(f"index : {data['indexes']}")
        # print(f"score:{output}")

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

        save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}")
        save_path.mkdir(exist_ok=True, parents=True)

        with open(f"{save_path}/NBestCrossBert_{mode}_result.json", "w") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=1)

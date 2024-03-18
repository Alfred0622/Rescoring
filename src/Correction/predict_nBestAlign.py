import sys
import os

sys.path.append("..")
sys.path.append("../..")
import json
import torch
import logging

from torch.utils.data import DataLoader

from utils.Datasets import get_dataset
from utils.CollateFunc import recogBatch
from src_utils.LoadConfig import load_config
from utils.PrepareModel import prepare_model
from models.nBestAligner.nBestTransformer import nBestAlignBart
from utils.Datasets import get_dataset
from utils.CollateFunc import nBestAlignBatch
from pathlib import Path

from jiwer import wer, cer
from tqdm import tqdm

assert (
    len(sys.argv) == 3
), "Usage: python ./predict_nBestAlign.py <checkpoint_path> <name>"

checkpoint = sys.argv[1]
save_name = sys.argv[2]

args, train_args, recog_args = load_config(f"./config/nBestAlign.yaml")

setting = "withLM" if args["withLM"] else "noLM"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, tokenizer = prepare_model(args["dataset"])
model = nBestAlignBart(args, train_args, tokenizer = tokenizer)

checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint['checkpoint'])
model = model.to(device)

model.show_param()
# exit()

if args["dataset"] in ["aishell", "aishell_nbest"]:
    recog_set = ["dev", "test"]
elif args["dataset"] in ["aishell2"]:
    recog_set = ["dev", "test_android", "test_ios", "test_mic"]
elif args["dataset"] in ["tedlium2"]:
    recog_set = ["dev", "test"]
elif args["dataset"] in ["csj"]:
    recog_set = ["dev", "eval1", "eval2", "eval3"]
elif args["dataset"] in ["librispeech"]:
    recog_set = ["dev_clean", "dev_other", "test_clean", "test_other"]

model.eval()
for task in recog_set:
    hyps = []
    refs = []
    top_hyps = []
    names = []
    data_json = f"../../data/{args['dataset']}/data/{setting}/{task}/data.json"

    with open(data_json) as f:
        data_json = json.load(f)

    dataset = get_dataset(
        data_json,
        dataset=args["dataset"],
        tokenizer=tokenizer,
        data_type="align",
        topk=int(args["nbest"]),
        sep_token=train_args["sep_token"],
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=recog_args["batch"],
        collate_fn=nBestAlignBatch,
    )

    result_dict = list()
    total_time = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader, ncols=80):
            token = data["input_ids"].to(device)
            mask = data["attention_mask"].to(device)

            output, elapsed_time = model.recognize(
                input_ids = token,
                attention_mask = mask,
                max_lens = 150,
                num_beams = 5
            )

            total_time += elapsed_time

            hyp_tokens = tokenizer.batch_decode(output, skip_special_tokens=True)

            # print('\n')
            # for hyp, top_hyp, ref, input_hyps in zip(hyp_tokens, data['top_hyp'],data['ref_text'], data['hyps_text']):
            #     for hyp_id, h in enumerate(input_hyps):
            #         print(f'input {hyp_id + 1}:{h}')
            #     print(f'hyp:{hyp}')
            #     print(f'top_hyp:{top_hyp}')
            #     print(f'ref:{ref}')
            #     print(f'=============================')

            hyps += hyp_tokens
            top_hyps += data['top_hyp']
            refs += data['ref_text']
            names += data['name']

            # for name, hyp, top_hyp, ref in zip(
            #     data["name"], hyp_tokens, data["top_hyp"], data["ref_text"]
            # ):
            #     hyps.append(hyp)
            #     top_hyps.append(top_hyp)
            #     refs.append(ref)
            #     names.append(name)

    print(f'average decode time : {total_time / len(data_json)}')
    for name, hyp, top_hyp, ref_token in zip(names, hyps, top_hyps, refs):
        corrupt_flag = "Missed"  # Missed only for debug purpose, to detect if there is any data accidentally ignored
        if top_hyp == ref_token:
            if hyp != ref_token:
                corrupt_flag = "Totally_Corrupt"
            else:
                corrupt_flag = "Remain_Correct"

        else:
            if hyp == ref_token:
                corrupt_flag = "Totally_Improve"

            else:
                top_wer = wer(ref_token, top_hyp)
                rerank_wer = wer(ref_token, hyp)
                if top_wer < rerank_wer:
                    corrupt_flag = "Partial_Corrupt"
                elif top_wer == rerank_wer:
                    corrupt_flag = "Neutral"
                else:
                    corrupt_flag = "Partial_Improve"
        # print(f'Corrupt Flag:{corrupt_f}')

        result_dict.append(
            {
                "name": name,
                "hyp": hyp_tokens,
                "ref": ref_token,
                "top_hyp": top_hyp,
                "check1": "Correct" if hyp_tokens == ref_token else "Wrong",
                "check2": corrupt_flag,
            }
        )

    print(f'len of hyps:{len(hyps)}')
    print(f'len of refs:{len(refs)}')
    print(f"hyp:{hyps[-1]}")
    print(f"org:{top_hyps[-1]}")
    print(f"ref:{refs[-1]}")

    if args["dataset"] in ["aishell", "aishell2", "csj", "aishell_nbest"]:
        print(f"{args['dataset']} {setting} {task} -- ORG CER = {wer(refs, top_hyps)}")
        print(f"{args['dataset']} {setting} {task} -- CER = {wer(refs, hyps)}")
    elif args["dataset"] in ["tedlium2", "librispeech"]:
        print(f"{args['dataset']} {setting} {task} -- ORG WER = {wer(refs, top_hyps)}")
        print(f"{args['dataset']} {setting} {task} -- WER = {wer(refs, hyps)}")

    save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}/")
    save_path.mkdir(parents=True, exist_ok=True)

    with open(f"{save_path}/{args['nbest']}Align_result.json", "w") as dest:
        json.dump(result_dict, dest, ensure_ascii=False, indent=1)

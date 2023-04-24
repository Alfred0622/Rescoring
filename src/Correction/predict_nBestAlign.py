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
from models.nBestAligner.nBestTransformer import nBestTransformer
from utils.Datasets import  get_dataset
from utils.CollateFunc import nBestAlignBatch
from pathlib import Path

from jiwer import wer, cer
from tqdm import tqdm

checkpoint = sys.argv[1]

args, train_args, recog_args = load_config(f'./config/nBestAlign.yaml')

setting = 'withLM' if args['withLM'] else 'noLM'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, tokenizer = prepare_model(
    args['dataset']
)

model = nBestTransformer(
    nBest=args['nbest'],
    device=device,
    lr=float(train_args["lr"]),
    align_embedding=train_args["align_embedding"],
    dataset = args['dataset']
)

checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint)

if (args['dataset'] in ['aishell', 'aishell_nbest']):
    recog_set = ['dev', 'test']
elif (args['dataset'] in ['aishell2']):
    recog_set = ['dev_ios', 'test_android', 'test_ios', 'test_mic']
elif (args['dataset'] in ['tedlium2']):
    recog_set = ['dev', 'test']
elif (args['dataset'] in ['csj']):
    recog_set = ['dev', 'eval1', 'eval2', 'eval3']


for task in recog_set:
    hyps = []
    refs = []
    top_hyps = []
    data_json = f"./data/{args['dataset']}/{setting}/{task}/{args['nbest']}_align_token.json"

    with open(data_json) as f:
        data_json = json.load(f)
    
    dataset = get_dataset(data_json, tokenizer = tokenizer, data_type = 'align', topk = int(args['nbest']))

    dataloader = DataLoader(
        dataset = dataset,
        batch_size = recog_args['batch'],
        collate_fn = nBestAlignBatch,
        num_workers = 4
    )

    result_dict = list()

    for data in tqdm(dataloader, ncols = 80):
        token = data['input_ids'].to(device)
        mask = data['attention_mask'].to(device)
        
        output = model.recognize(
            input_id = token,
            attention_mask = mask,
            max_lens = 50
        )

        for hyp, top_hyp ,ref in zip(output, data['top_hyp'],data['refs']):
            hyp_tokens = tokenizer.decode(hyp, skip_special_tokens = True)
            ref_tokens = tokenizer.decode(ref, skip_special_tokens = True)
            
            # hyp_tokens = hyp_tokens.replace('[SEP]', "")
            # hyp_tokens = hyp_tokens.replace('<\s>', "")
            if (args['dataset'] in ['aishell_nbest']):
                top_hyp = [h for h in top_hyp]
                top_hyp = " ".join(top_hyp)
            hyps.append(hyp_tokens)
            top_hyps.append(top_hyp)
            refs.append(ref_tokens)

            corrupt_flag = "Missed" # Missed only for debug purpose, to detect if there is any data accidentally ignored

            if (top_hyp == ref_tokens):
                if (hyp_tokens != ref_tokens):
                    corrupt_flag = "Totally_Corrupt"
                else:
                    corrupt_flag = "Remain_Correct"

            else :
                if (hyp_tokens == ref_tokens):
                    corrupt_flag = "Totally_Improve"

                else:
                    top_wer = wer(ref_tokens, top_hyp)
                    rerank_wer = wer(ref_tokens, hyp_tokens)
                    if (top_wer < rerank_wer):
                        corrupt_flag = "Partial_Corrupt"
                    elif (top_wer == rerank_wer):
                        corrupt_flag = "Neutral"
                    else:
                        corrupt_flag = "Partial_Improve"
            # print(f'Corrupt Flag:{corrupt_f}')
                
    
            result_dict.append(
                {
                    "hyp": hyp_tokens,
                    "ref": ref_tokens,
                    "top_hyp": top_hyp,
                    "check1": "Correct" if hyp_tokens == ref_tokens else "Wrong",
                    "check2": corrupt_flag
                }
            )
                    

    print(f"hyp:{hyps[-1]}")
    print(f"top_hyp:{top_hyp}")
    print(f"ref:{refs[-1]}")

    if (args['dataset'] in ['aishell', 'aishell2', 'csj', 'aishell_nbest']):
        print(f"{args['dataset']} {setting} {task} -- ORG CER = {wer(refs, top_hyps)}")
        print(f"{args['dataset']} {setting} {task} -- CER = {wer(refs, hyps)}")
    elif (args['dataset'] in ['tedlium2', 'librispeech']):
        print(f"{args['dataset']} {setting} {task} -- ORG WER = {wer(refs, top_hyps)}")
        print(f"{args['dataset']} {setting} {task} -- WER = {wer(refs, hyps)}")
    
    save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}/")
    save_path.mkdir(parents = True, exist_ok = True)

    with open(f"{save_path}/{args['nbest']}Align_result.json", 'w') as f:
        json.dump(result_dict, f, ensure_ascii = False, indent = 1)

import sys
import os
sys.path.append("..")
sys.path.append("../..")
import json
import torch
import logging
from transformers import (
    AutoModelForSeq2SeqLM,
    BertTokenizer,
)

from torch.utils.data import DataLoader

from utils.Datasets import get_dataset
from utils.CollateFunc import recogBatch
from src_utils.LoadConfig import load_config
from utils.PrepareModel import prepare_model

from pathlib import Path

from jiwer import wer, cer
from tqdm import tqdm

if (len(sys.argv) != 3):
    raise AssertionError("Usage: python predict_nbestAlignConcat.py <mode> <checkpoint_path>")

task_name = sys.argv[1]
checkpoint_path = sys.argv[2]

use_train = False

def predict(model, tokenizer, loader):
    result = {}
    model.eval()

    print('predict')

    loop = tqdm(loader, total = len(loader), ncols=100)
    for batch in loop:
        name = batch["name"]
        input_ids = batch["input_ids"]
        # print(f'input_ids:{input_ids}')
        attention_mask = batch["attention_mask"]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # print(f'attention_mask:{attention_mask}')

        with torch.no_grad():
            # output = model.predict()
            output = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_length = 150,
                num_beams = 5,
                # early_stopping = True
            )

            # print(f'output:{output}')

            output = tokenizer.batch_decode(output, skip_special_tokens = True)
            ref_list = tokenizer.batch_decode(batch["labels"], skip_special_tokens = True)

            # for hyp, ref in zip(output, ref_list):
            #     print(f'hyp:{hyp}\n ref:{ref}')

            for single_name, pred, ref in zip(name, output, ref_list):
                if (single_name not in result.keys()):
                    result[single_name] = dict()
                result[single_name]["hyp"] = "".join(pred).strip()
                result[single_name]["ref"] = "".join(ref).strip()

                # print(f'hyp:{result[single_name]["hyp"]}\n ref:{result[single_name]["ref"]}')
    
    return result

# Predict
if __name__ == '__main__':
    if (task_name == 'plain'):
        config_name = './config/nBestPlain.yaml'
    else:
        config_name = './config/Bart.yaml'
        topk = 1
    
    print(f'config:{config_name}')

    args, train_args, recog_args = load_config(config_name)
    setting =  'withLM' if args['withLM'] else 'noLM'
    if (args['dataset'] == 'old_aishell'):
        setting = ""
    
    if (task_name == 'plain'):
        topk = args['nbest']
        print(f"sep_token:{train_args['sep_token']}")

    print(f'setting:{setting}')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if (use_train):
        scoring_set = ['train']
    elif (args['dataset'] in ["csj"]):
        scoring_set = ['dev', 'eval1', 'eval2', 'eval3']
    elif (args['dataset'] in ["aishell2"]):
        scoring_set = ["dev_ios", 'test_ios', 'test_mic', 'test_android']
    elif (args['dataset'] in ['librispeech']):
        scoring_set = ['dev_clean', 'dev_other', 'test_clean', 'test_other']
    else:
        scoring_set = ['dev', 'test']

    model, tokenizer = prepare_model(args['dataset'])
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)
    
    for data_name in scoring_set:
        print(f"recognizing:{data_name}")
        with open(f"../../data/{args['dataset']}/data/{setting}/{data_name}/data.json") as f:
            data_json = json.load(f)

        dataset = get_dataset(data_json, tokenizer, data_type = train_args['data_type'], sep_token=train_args['sep_token'] ,topk = topk, for_train = False)

        dataloader = DataLoader(
            dataset,
            collate_fn = recogBatch,
            batch_size = recog_args['batch'],
            num_workers = 5
        )

        output = predict(model, tokenizer, dataloader)
        if (not os.path.exists(f"./data/{args['dataset']}/{setting}/{data_name}/{args['nbest']}{task_name}/")):
            os.makedirs(f"./data/{args['dataset']}/{setting}/{data_name}/{args['nbest']}{task_name}/")
        with open(
            f"./data/{args['dataset']}/{setting}/{data_name}/{args['nbest']}{task_name}/correct_data.json",
            'w', 
            encoding = 'utf-8'
        ) as f:
            json.dump(output, f, ensure_ascii = False, indent = 4)
        
        ref = []
        hyp = []
        top_1_hyp = []

        result_dict = []

        for data in data_json:
            top_1_hyp.append(data['hyps'][0])
            ref.append(output[data['name']]['ref'])
            hyp.append(output[data['name']]['hyp'])

            corrupt_flag = "Missed"

            if (data['hyps'][0] == output[data['name']]['ref']):
                if (output[data['name']]['hyp'] != output[data['name']]['ref']):
                    corrupt_flag = "Totally_Corrupt"
                else:
                    corrupt_flag = "Remain_Correct"

            else:
                if (output[data['name']]['hyp'] == output[data['name']]['ref']):
                    corrupt_flag = "Totally_Improve"
                else:
                    top_wer = wer(output[data['name']]['ref'], data['hyps'][0])
                    rerank_wer = wer(output[data['name']]['ref'], output[data['name']]['hyp'])
                    if (top_wer < rerank_wer):
                        corrupt_flag = "Partial_Corrupt"
                    elif (top_wer == rerank_wer):
                        corrupt_flag = "Remain_Incorrect"
                    else:
                        corrupt_flag = "Partial_Improve"
    
            result_dict.append(
                {
                    "hyp": output[data['name']]['hyp'],
                    "ref": output[data['name']]['ref'],
                    "top_hyp": data['hyps'][0],
                    "check1": "Correct" if output[data['name']]['hyp'] == output[data['name']]['ref'] else "Wrong",
                    "check2": corrupt_flag
                }
            )
        
        
        save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{data_name}/")
        save_path.mkdir(parents = True, exist_ok = True)

        with open(f"{save_path}/{args['nbest']}_{task_name}_Correct_result.json", 'w') as f:
            json.dump(result_dict, f, ensure_ascii = False, indent = 1)
        
        print(f'org_wer:{wer(ref, top_1_hyp)}')
        print(f'Correction wer:{wer(ref, hyp)}')
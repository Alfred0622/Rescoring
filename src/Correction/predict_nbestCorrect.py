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
from utils.LoadConfig import load_config
from utils.PrepareModel import prepare_model
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

    loop = tqdm(loader, total = len(loader))
    for batch in loop:
        name = batch["name"]
        input_ids = batch["input_ids"]
        # print(f'input_ids:{input_ids}')
        attention_mask = batch["attention_mask"]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # print(f'attention_mask:{attention_mask}')

        with torch.no_grad():
            output = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_length = 150,
                num_beams = 5,
                early_stopping = True
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
    
    return result

# Predict
if __name__ == '__main__':
    if (task_name == 'align_concat'):
        config_name = './config/nBestPlain.yaml'
        topk = 1
    else:
        config_name = './config/Bart.yaml'
        topk = 1

    args, train_args, recog_args = load_config(config_name)
    setting =  'withLM' if args['withLM'] else 'noLM'
    if (args['dataset'] == 'old_aishell'):
        setting = ""
    
    print(f'setting:{setting}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if (use_train):
        scoring_set = ['train']
    elif (args['dataset'] in ["csj"]):
        scoring_set = ['dev', 'eval1', 'eval2', 'eval3']
    elif (args['dataset'] in ["aishell2"]):
        scoring_set = ["dev_ios", 'test_ios', 'test_mic', 'test_android']
    else:
        scoring_set = ['test', 'dev']

    model, tokenizer = prepare_model(args['dataset'])
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)
    
    for data_name in scoring_set:
        print(f"recognizing:{data_name}")
        with open(f"../../data/{args['dataset']}/data/{setting}/{data_name}/data.json") as f:
            data_json = json.load(f)
        
        dataset = get_dataset(data_json, tokenizer, topk = topk, for_train = False)

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
        for data in data_json:
            top_1_hyp.append(data['hyps'][0])
            ref.append(output[data['name']]['ref'])
            hyp.append(output[data['name']]['hyp'])
        
        print(f'org_wer:{wer(ref, top_1_hyp)}')
        print(f'Correction wer:{wer(ref, hyp)}')
        
        # for data in output.keys():

        #     ref.append(output[data]['ref'])
        #     hyp.append(output[data]['hyp'])
    
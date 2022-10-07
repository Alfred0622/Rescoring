import torch
import json
import logging
from transformers import BertTokenizer, BartTokenizer
import os
from tqdm import tqdm

setting = ["withLM", "noLM"]
dataset = ["dev", "test","train"]  # train
model_name = "bert"
task = "Correction"
nbest = 50
data_name = 'aishell'
addidtional_name = "3align_concat"


if (data_name in ['aishell', 'aishell2']):
    if model_name == "bart":
        tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    elif model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
elif (data_name in ['tedlium2', 'librispeech']):
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

for d in dataset:
    for s in setting:
        print(f"{d}:{s}")
        file_name = f"./data/{data_name}/{s}/{d}/{addidtional_name}/data.json"
        with open(file_name) as f:
            data_json = json.load(f)
        
        tokenized_data = list()
        for hyp, ref in zip(data_json['token'], data_json['ref']):
            hyp_str = str()
            for h in hyp:
                hyp_str += h
            
            input_ids, _,attention_mask = tokenizer(hyp_str).values()
            tokenized_data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": tokenizer(ref)["input_ids"]
                }
            )
            
        
        # tokenized_data['ref'] = data_json['ref']

        with open(f"./data/{data_name}/{s}/{d}/{addidtional_name}/token.json", 'w') as f:
            json.dump(tokenized_data, f, ensure_ascii = False, indent = 4)
        print(tokenized_data[0])


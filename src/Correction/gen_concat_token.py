import sys
import torch
import json
import logging
from transformers import BertTokenizer, BartTokenizer, BertTokenizerFast
import os
from tqdm import tqdm

data_name = sys.argv[1]
model_name = sys.argv[2]
data_type = sys.argv[3]

setting = ['noLM', 'withLM']
if (data_name == 'csj'):
    data = ['train', 'dev', 'eval1', 'eval2', 'eval3']
else:
    data = ['train', 'dev', 'test']

assert (model_name.strip().upper() in ['BERT', 'BART', 'GPT2']), "model only assist BERT, BART, GPT2 only"

# set tokenizer
if (data_name in ['aishell', 'aishell2']):
    if (model_name.strip().upper() == 'BERT'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    elif (model_name.strip().upper() == 'BART'):
        tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
    elif (model_name.strip().upper() == 'GPT2'):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
elif (data_name in ['tedlium2', 'librispeech']):
    if (model_name.strip().upper() == 'BERT'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif (data_name in ['csj']):
    pass

for s in setting:
    for d in data:
        print(f'{d}:{s}')
        json_file = f"./data/{data_name}/{s}/{d}/{data_type}/data.json"
        w_json = f"./data/{data_name}/{s}/{d}/{data_type}/token.json"
        with open(json_file, "r") as f, open(w_json, "w") as fw:
            j = json.load(f)
            ids = []
            
            for seq in tqdm(j["token"]):
                token = ["[CLS]"] + seq + ["[SEP]"]
                ids.append(tokenizer.convert_tokens_to_ids(token))
            
            ref_ids = []
            for ref in j["ref"]:
                ref = tokenizer.tokenize("[CLS]" + ref + "[SEP]")
                ref_ids.append(tokenizer.convert_tokens_to_ids(ref))
            
            write_data = {
                "token": ids,
                "ref_token": ref_ids,
                "ref": j["ref"]
            }

            json.dump(write_data, fw, ensure_ascii=False, indent=4)
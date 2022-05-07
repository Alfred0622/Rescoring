import torch
import json
import logging
from transformers import BertTokenizer, BartTokenizer

FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, filename='gen_token.log', filemode='w', format=FORMAT)



dataset = ['train', 'dev', 'test'] # train
model_name = 'bart'


if (model_name == 'bart'):
    tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
elif (model_name == 'bert'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

for d in dataset:
    print(d)
    json_file = f'./data/aishell_{d}/dataset.json'
    w_json = f'./data/aishell_{d}/{model_name}_token/token.json'
    with open(json_file, 'r') as f, open(w_json, 'w') as fw :
        j = json.load(f)
        for i, element in enumerate(j):
            ids = []
            seg = []
            for seq in element['token']:
                tokens = tokenizer.tokenize(seq)
                ids.append(tokenizer.convert_tokens_to_ids(tokens))
                seg.append([0] * len(ids[-1]))
                # logging.warning(seg)
            element['name'] = f'{d}_{i}'
            element['token'] = ids
            element['segment'] = seg
            
            ref_token = tokenizer.tokenize(element['ref'])
            element['ref_token'] = tokenizer.convert_tokens_to_ids(ref_token)
            element['ref_seg'] = [0] * len(element['ref_token'])
        json.dump(j, fw, ensure_ascii=False, indent = 4)
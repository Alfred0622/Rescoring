import torch
import json
import logging
from transformers import BertTokenizer

FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, filename='run.log', filemode='w', format=FORMAT)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

dataset = ['dev', 'test'] # train

for d in dataset:
    json_file = f'./data/aishell_{d}/dataset.json'
    w_json = f'./data/aishell_{d}/token.json'
    with open(json_file, 'r') as f, open(w_json, 'w') as fw :
        j = json.load(f)
        for element in j:
            ids = []
            seg = []
            for seq in element['token']:
                tokens = tokenizer.tokenize(seq)
                ids.append(tokenizer.convert_tokens_to_ids(tokens))
                seg.append([0] * len(ids[-1]))
                # logging.warning(seg)
            element['token'] = ids
            element['segment'] = seg
        json.dump(j, fw, ensure_ascii=False)
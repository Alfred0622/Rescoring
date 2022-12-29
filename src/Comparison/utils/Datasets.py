import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json

class concatTrainerDataset(Dataset):
    def __init__(self, nbest_list):
        self.data = nbest_list
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def get_dataset(data_json, tokenizer, is_token = False):
    if (is_token):
        return concatTrainerDataset(data_json), None
    else:
        data_list = list()
        for data in tqdm(data_json):
            input_ids, token_type_ids, attention_mask = tokenizer(data['hyp1'], data['hyp2']).values()
            label = data['label']
            data_list.append(
                {
                    "input_ids":input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "labels": label
                }
            )
        return concatTrainerDataset(data_list), data_list

def get_alsemDataset(data_json, tokenizer):
    data_list = list()
    for data in tqdm(data_json):
        input_ids, token_type_ids, attention_mask = tokenizer(data['hyp1'], data['hyp2']).values()
        label = data['label']
        am_score = torch.tensor(data['am_score'])
        ctc_score = torch.tensor(data['ctc_score'])
        lm_score = torch.tensor(data['lm_score'])
        data_list.append(
            {
                "input_ids":input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": label,
                "am_score": am_score,
                "ctc_score": ctc_score,
                "lm_score": lm_score
            }
        )
    return concatTrainerDataset(data_list)

def get_recogDataset(data_json, tokenizer):
    data_list = list()
    for data in tqdm(data_json):    
        name  = data['name']
        # print(f"{len(data['hyp'])}")
        for hyps in data['hyps']:
            input_id, token_type_id, mask = tokenizer(
                hyps['hyp1'], hyps['hyp2']
            ).values() 
            pair = hyps['pair']
            data_list.append(
                {
                    "name": name,
                    "input_ids": input_id,
                    "token_type_ids": token_type_id,
                    "attention_mask": mask,
                    "pair": pair,
                }
            )

    return concatTrainerDataset(data_list)

def get_BertAlsemrecogDataset(data_json, tokenizer):
    data_list = list()
    for data in tqdm(data_json):    
        name  = data['name']
        # print(f"{len(data['hyp'])}")
        for hyps in data['hyp']:
            input_id, token_type_id, mask = tokenizer(
                hyps['hyp1'], hyps['hyp2']
            ).values() 
            pair = hyps['pair']
            am_scores = hyps['am_score']
            ctc_scores = hyps['ctc_score']
            lm_scores = hyps['lm_score']

            data_list.append(
                {
                    "name": name,
                    "input_ids": input_id,
                    "token_type_ids": token_type_id,
                    "attention_mask": mask,
                    "am_score": am_scores,
                    "ctc_score": ctc_scores,
                    "lm_score": lm_scores,
                    "pair": pair,
                }
            )

    return concatTrainerDataset(data_list)
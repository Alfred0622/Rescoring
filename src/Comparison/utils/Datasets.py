import sys
sys.path.append("../")
sys.path.append("../..")
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from src_utils.preprocess import preprocess_string

class concatTrainerDataset(Dataset):
    def __init__(self, nbest_list):
        self.data = nbest_list
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def get_dataset(data_json, dataset ,tokenizer, is_token = False):
    if (is_token):
        return concatTrainerDataset(data_json), None
    else:
        data_list = list()
        for data in tqdm(data_json, ncols = 60):
            hyp1 = preprocess_string(data['hyp1'], dataset)
            hyp2 = preprocess_string(data['hyp2'], dataset)
            input_ids, token_type_ids, attention_mask = tokenizer(hyp1, hyp2, max_length = 512, truncation = True).values()
            label = data['label']
            data_list.append(
                {
                    "input_ids":input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "labels": label
                }
            )
        
        data_list = sorted(data_list, key = lambda x : len(x['input_ids']))
        return concatTrainerDataset(data_list), data_list

def get_alsemDataset(data_json, dataset,tokenizer, for_train = True):
    data_list = list()
    for i, data in enumerate(tqdm(data_json, ncols = 100)):
        hyp1 = preprocess_string(data['hyp1'], dataset)
        hyp2 = preprocess_string(data['hyp2'], dataset)
        input_ids, token_type_ids, attention_mask = tokenizer(hyp1, hyp2).values()
        label = data['label']
        am_score = torch.tensor(data['am_score'])
        ctc_score = torch.tensor(data['ctc_score'])
        lm_score = torch.tensor(data['lm_score'])
        data_list.append(
            {
                "input_ids":input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": label if for_train else None,
                "am_score": am_score,
                "ctc_score": ctc_score,
                "lm_score": lm_score
            }
        )

    data_list = sorted(data_list, key = lambda x : len(x['input_ids']))
    return concatTrainerDataset(data_list)

def get_recogDataset(data_json, dataset,tokenizer):
    data_list = list()
    for i, data in enumerate(tqdm(data_json, ncols = 100)):

        name  = data['name']
        # print(f"{len(data['hyp'])}")
        for hyps in data['hyps']:
            hyp1 = preprocess_string(hyps['hyp1'], dataset)
            hyp2 = preprocess_string(hyps['hyp2'], dataset)
            input_id, token_type_id, mask = tokenizer(hyp1, hyp2).values() 
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

def get_recogDatasetFromRaw(data_json, dataset, tokenizer):
    data_list = list()
    
    if (isinstance(data_json, dict)):
        for i, key in enumerate(tqdm(data_json.keys(), ncols = 100)):
            name = data_json[key]['name']

            for hyp in data_json[key]['hyps'][:topk]
            
            
        

def get_BertAlsemrecogDataset(data_json, dataset, tokenizer):
    data_list = list()
    for data in tqdm(data_json, ncols = 100):    
        name  = data['name']

        for hyps in data['hyps']:
            hyp1 = preprocess_string(hyps['hyp1'], dataset)
            hyp2 = preprocess_string(hyps['hyp2'], dataset)
            # print(f"hyp1:{hyps['hyp1']}")
            # print(f"hyp1 after preprocess:{hyp1}")

            # print(f"hyp2:{hyps['hyp2']}")
            # print(f"hyp2 after preprocess:{hyp2}")
            input_id, token_type_id, mask = tokenizer(
                hyp1, hyp2
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
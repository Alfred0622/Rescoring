import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class concatTrainerDataset(Dataset):
    def __init__(self, nbest_list):
        self.data = nbest_list
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def get_dataset(data_json, tokenizer):
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
    return concatTrainerDataset(data_list)

def get_recogDataset(data_json, tokenizer):
    data_list = list()
    for data in tqdm(data_json):    
        name  = data['name']
        # print(f"{len(data['hyp'])}")
        for hyps in data['hyp']:
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
                    "pair": pair
                }
            )

    return concatTrainerDataset(data_list)

class concatDataset(Dataset):
    # Dataset for BertComparision
    # Here, data will have
    # 1.  [CLS] seq1 [SEP] seq2
    # 2.  labels
    def __init__(self, nbest_list, nbest=10):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list

    def __getitem__(self, idx):
        return (
            self.data[idx]["token"],
            self.data[idx]['label']
        )

    def __len__(self):
        return len(self.data)

class compareRecogDataset(Dataset):
    def __init__(self, nbest_list, nbest=10):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list

    def __getitem__(self, idx):
        return (
            self.data[idx]['name'],
            self.data[idx]["token"],
            self.data[idx]['pair'],
            self.data[idx]['text'],
            self.data[idx]['score'],
            self.data[idx]['err'],
            self.data[idx]['ref'],
        )

    def __len__(self):
        return len(self.data)

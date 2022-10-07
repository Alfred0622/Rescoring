import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class correctTrainerDataset(Dataset):
    def __init__(self, nbest_list):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def get_dataset(data_json, tokenizer, topk, data_type = 'single',for_train = True):
    """
    data_type: str. can be "single" or "concat"
    """
    assert(isinstance(topk, int))

    if (topk < 1):
        topk = 1
    data_list = list()
    if (for_train):
        for data in tqdm(data_json):
            label = tokenizer(data['ref'])["input_ids"]
            for i in range(topk):
                input_ids, token_type_id, attention_mask = tokenizer(data['hyp'][i]).values()
                data_list.append(
                    {
                        "input_ids":input_ids,
                        "attention_mask": attention_mask,
                        "labels": label
                    }
            )
        return correctTrainerDataset(data_list)
    else:
        for data in tqdm(data_json):
            name = data['name']
            input_ids, token_type_id, attention_mask = tokenizer(data['hyp'][0]).values()
            label = tokenizer(data['ref'])["input_ids"]

            data_list.append(
                {
                    "name": name,
                    "input_ids":input_ids,
                    "attention_mask": attention_mask,
                    "labels": label
                }
            )
        return correctTrainerDataset(data_list)

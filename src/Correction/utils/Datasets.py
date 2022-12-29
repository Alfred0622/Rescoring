import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.nBestAligner.nBestAlign import align, alignNbest

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

def get_dataset(data_json, tokenizer, topk, sep_token = '[SEP]', data_type = 'single', for_train = True):
    """
    data_type: str. can be "single" or "concat"
    """
    assert(isinstance(topk, int))

    if (topk < 1):
        topk = 1
    data_list = list()

    if (for_train):
        if (data_type == 'single'):
            for data in tqdm(data_json):
                label = tokenizer(data['ref'])["input_ids"]

                for hyp in data['hyps'][:topk]:
                    output = tokenizer(hyp)
                    if ('token_type_ids' in output.keys()):
                        input_ids, _ , attention_mask = output.values()
                    else:
                        input_ids, attention_mask = output.values()
                    data_list.append(
                        {
                            "input_ids":input_ids,
                            "attention_mask": attention_mask,
                            "labels": label
                        }
                )

        elif (data_type == 'concat'):
            if (sep_token == '[SEP]'):
                sep_token = tokenizer.sep_token
            for data in tqdm(data_json):
                label = tokenizer(data['ref'])["input_ids"]
                concat_str = str()
                for i in range(topk):
                    if (i == topk - 1):
                        concat_str += data['hyps'][i] + tokenizer.sep_token
                    else:
                        concat_str += data['hyps'][i] + sep_token
                
                output = tokenizer(concat_str)

                if ('token_type_id' in output.keys() ):
                    input_ids, _ , attention_mask = output.values
                else:
                    input_ids, attention_mask = output.values
                
                data_list.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": label
                    }
                )
            
        elif (data_type == 'align'):
            for data in tqdm(data_json):
                label = tokenizer(data['ref'])["input_ids"]

        return correctTrainerDataset(data_list)

    else:
        if (data_type == 'single'):
            for data in tqdm(data_json):
                name = data['name']
                output = tokenizer(data['hyps'][0])
                if ('token_type_ids' in output.keys()):
                    input_ids, _ , attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
                label = tokenizer(data['ref'])["input_ids"]

                data_list.append(
                    {
                        "name": name,
                        "input_ids":input_ids,
                        "attention_mask": attention_mask,
                        "labels": label
                    }
                )

        elif (data_type == 'concat'):
            if (sep_token == '[SEP]'):
                sep_token = tokenizer.sep_token

            for data in tqdm(data_json):
                name = data['name']
                concat_str = str()
                for i in range(topk):
                    if (i == topk - 1):
                        concat_str += (data['hyps'][i]) + tokenizer.sep_token
                    else:
                        concat_str += (data['hyps'][i]) + sep_token
                
                output = tokenizer(concat_str)

                if ('token_type_ids' in output.keys()):
                    input_ids, _, attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
                
            data_list.append(
                {
                    "name": name,
                    "input_ids":input_ids,
                    "attention_mask": attention_mask,
                    "labels": label
                }
            )
                    
        return correctTrainerDataset(data_list)

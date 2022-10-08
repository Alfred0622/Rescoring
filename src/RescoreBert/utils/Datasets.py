import torch
from torch.utils.data import Dataset, DataLoader

class LM_Dataset(Dataset):
    def __init__(self, nbest_list):
        self.data = nbest_list
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def get_Dataset(data_json, tokenizer, dataset,for_train = True, topk = 20):
    data_list = list()

    if (for_train):
        for data in data_json:
            if (dataset in ['aishell', 'aishell2', 'old_aishell']):
                input_ids, _, attention_mask = tokenizer(data['ref']).values()
            elif (dataset in ['tedlium2', 'librispeech']):
                input_ids, attention_mask = tokenizer(data['ref']).values()

            input_ids = torch.tensor(input_ids, dtype = torch.int32)
            attention_mask = torch.tensor(attention_mask, dtype = torch.int32)
            data_list.append(
                {
                    "input_ids": input_ids,
                    # "attention_mask": attention_mask,
                    # "labels": input_ids.clone()
                }
            )

        return LM_Dataset(data_list)

    else:
        for data in data_json:
            if (topk > len(data["hyp"])):
                nbest = len(data["hyp"])
            else: nbest = topk
            name = data['name']
            for i, hyp in enumerate(data['hyp'][:nbest]):
                if (dataset in ['aishell', 'aishell2']):
                    input_ids, _, attention_mask = tokenizer(hyp).values()
                elif (dataset in ['tedlium2', 'librispeech']):
                    input_ids, attention_mask = tokenizer(hyp).values()
            
                data_list.append(
                    {
                        "name": name,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "index": i
                    }
                )

        return LM_Dataset(data_list)

def get_mlm_dataset(data_list, tokenizer, topk = 20):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id

    data_list = list()

    for data in data_list:
        name = data['name']
        for i, hyp in enumerate(data['hyp'][:topk]):
            input_ids, token_type_ids, attention_mask = tokenizer(hyp)
            for j, ids in enumerate(input_ids):
                temp_input = input_ids.copy()
                if (ids in [bos_id, eos_id]):
                    continue
                temp_input[i] = mask_id

                data_list.append(
                    {
                        "name": name,
                        "input_ids": torch.tensor(temp_input),
                        "attention_mask": torch.tensor(attention_mask),
                        "mask_token": ids, # the token that is masked
                        "nbest": i,   # Nbest index
                        "index": j    # In which position
                    }
                )
        
    return LM_Dataset(data_list)

class adaptionDataset(Dataset):
    # Use only for domain adaption in MLM bert
    # Only use groundtruth
    def __init__(self, nbest_list):
        self.data = nbest_list

    def __getitem__(self, idx):
        return self.data[idx]["ref_token"]

    def __len__(self):
        return len(self.data)

class pllDataset(Dataset):
    # Training dataset
    def __init__(self, nbest_list, nbest = 50):
        """
        nbest_list: list()
        """
        self.data = nbest_list
        self.nbest = nbest

    def __getitem__(self, idx):
        if (self.nbest <= len(self.data[idx]["token"])):
            return (
            
                self.data[idx]["token"][: self.nbest],
                self.data[idx]["text"][: self.nbest],
                self.data[idx]["score"][: self.nbest],
                self.data[idx]["err"][: self.nbest],
                self.data[idx]["pll"][: self.nbest],
                self.data[idx]["name"]
            )
        else:
            return (
            
                self.data[idx]["token"],
                self.data[idx]["text"],
                self.data[idx]["score"],
                self.data[idx]["err"],
                self.data[idx]["pll"],
            )
        #    self.data[idx]['name'],\

    def __len__(self):
        return len(self.data)

class nBestDataset(Dataset):
    def __init__(self, nbest_list, nbest = 50):
        """
        nbest_list: list()
        """
        self.data = nbest_list
        self.nbest = nbest

    def __getitem__(self, idx):
        if (self.nbest <= len(self.data[idx]["token"])):
            return (
                self.data[idx]["token"][: self.nbest],
                self.data[idx]["text"][: self.nbest],
                self.data[idx]["score"][: self.nbest],
                self.data[idx]["err"][: self.nbest],
            )
        else:
            return (
                self.data[idx]["token"],
                self.data[idx]["text"],
                self.data[idx]["score"],
                self.data[idx]["err"],
            )
        #    self.data[idx]['name'],\
    def __len__(self):
        return len(self.data)

class rescoreDataset(Dataset):
    def __init__(self, nbest_list, nbest=10):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list
        self.nbest = nbest

    def __getitem__(self, idx):
       if (self.nbest <= len(self.data[idx]["token"])):
            return (
                self.data[idx]["name"],
                self.data[idx]["token"][: self.nbest],
                self.data[idx]["text"][: self.nbest],
                self.data[idx]["score"][: self.nbest],
                self.data[idx]["ref"],
                self.data[idx]["err"][: self.nbest],
            )
       else:
            return (
                self.data[idx]["name"],
                self.data[idx]["token"],
                self.data[idx]["text"],
                self.data[idx]["score"],
                self.data[idx]["ref"],
                self.data[idx]["err"],
            )

    def __len__(self):
        return len(self.data)


class rescoreComplexDataset(Dataset):
    # return with am, lm and ctc score seperately 
    def __init__(self, nbest_list, nbest=10):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list
        self.nbest = nbest

    def __getitem__(self, idx):
       if (self.nbest <= len(self.data[idx]["token"])):
            return (
                self.data[idx]["name"],
                self.data[idx]["token"][: self.nbest],
                self.data[idx]["text"][: self.nbest],
                self.data[idx]["am_score"][: self.nbest],
                self.data[idx]["ctc_score"][: self.nbest],
                self.data[idx]["lm_score"][: self.nbest],
                self.data[idx]["ref"],
                self.data[idx]["err"][: self.nbest],
            )
       else:
            return (
                self.data[idx]["name"],
                self.data[idx]["token"],
                self.data[idx]["text"],
                self.data[idx]["am_score"],
                self.data[idx]["lm_score"],
                self.data[idx]["ctc_score"],
                self.data[idx]["ref"],
                self.data[idx]["err"],
            )

    def __len__(self):
        return len(self.data)

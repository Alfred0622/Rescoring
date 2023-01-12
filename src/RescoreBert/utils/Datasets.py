import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class LM_Dataset(Dataset):
    def __init__(self, nbest_list):
        self.data = nbest_list
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
        
def get_Dataset(data_json, tokenizer, dataset, lm = "CLM",for_train = True, topk = 20, jp_split = True):
    # jp_split: remove space from hyps or refs of jp dataset
    data_list = list()
    
    print(f'dataset:{dataset}')
    print(f'lm:{lm}')

    if (for_train):
        if (lm == "MLM"):
            for data in tqdm(data_json, ncols = 80):

                if (dataset in ['csj']):
                    if (jp_split):
                        ref = "".join(data['ref'].split())
                    else:
                        ref = data['ref']
                        
                else:
                    ref = data['ref']
                output = tokenizer(ref)

                if ('token_type_ids' in output.keys()):
                    input_ids, _, attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()

                input_ids = torch.tensor(input_ids, dtype = torch.int32)
                attention_mask = torch.tensor(attention_mask, dtype = torch.int32)
                data_list.append(
                    {
                        "input_ids": input_ids,
                        # "attention_mask": attention_mask,
                        # "labels": input_ids.clone()
                    }
                )

        elif (lm in ["CLM", "CLM_char"]):
            tokenizer.bos_token = tokenizer.cls_token if tokenizer.bos_token is None else tokenizer.bos_token
            tokenizer.eos_token = tokenizer.sep_token if tokenizer.eos_token is None else tokenizer.eos_token
            for data in tqdm(data_json, ncols = 80):
                
                if (dataset in ['csj'] and jp_split):
                    ref = "".join(data['ref'].split())
                elif (dataset in ['tedlium2', 'tedlium2_conformer', 'librispeech']):
                    ref = data['ref'] + "."
                else:
                    ref = data['ref']

                if (dataset in ['aishell', 'aishell2']):
                    pass
                else:
                    ref = f'{tokenizer.bos_token} {ref} {tokenizer.eos_token}'

                output = tokenizer(ref)

                if ('token_type_ids' in output.keys()):
                    input_ids, _ , attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
        
                if (len (input_ids) <= 1): continue
                input_ids = torch.tensor(input_ids, dtype = torch.int32)
                data_list.append(
                    {
                        "input_ids": input_ids,
                    }
                )
            print(f'# num of Dataset:{len(data_list)}')

        return LM_Dataset(data_list)

    else:
        for data in tqdm(data_json, ncols = 80):
            if (topk > len(data["hyps"])):
                nbest = len(data["hyps"])
            else: nbest = topk
            name = data['name']
            for i, hyp in enumerate(data['hyps'][:nbest]):
                if (dataset in ['csj'] and jp_split):
                    hyp = "".join(hyp.split())
                output = tokenizer(hyp)
                if ('token_type_ids' in output.keys()):
                    input_ids, _, attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
            
                data_list.append(
                    {
                        "name": name,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "index": i
                    }
                )

        return LM_Dataset(data_list)

def get_mlm_dataset(data_json, tokenizer,  dataset, topk = 50, jp_split = True):

    bos_id = tokenizer.cls_token_id if tokenizer.bos_token_id is None else tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.eos_token_id is None else tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    data_list = list()

    assert (bos_id is not None and eos_id is not None and mask_id is not None), f"{bos_id}, {eos_id}, {mask_id}"

    for data in tqdm(data_json, ncols = 80):
        if (topk > len(data["hyps"])):
            nbest = len(data["hyps"])
        else: nbest = topk
        name = data['name']

        for i, hyp in enumerate(data['hyps'][:nbest]):
            if (dataset in ['csj'] and jp_split):
                hyp = "".join(hyp.split())
            output = tokenizer(hyp)
            if ('token_type_ids' in output.keys()):
                input_ids, _, attention_mask = output.values()
            else:
                input_ids, attention_mask = output.values()

            if (len(input_ids) == 2):
                for j, ids in enumerate(input_ids):
                    temp_ids = output['input_ids'].copy()
                    masked_token = temp_ids[j]
                    temp_ids[j] = tokenizer.mask_token_id
                    data_list.append(
                        {
                            "name": name,
                            "input_ids": torch.tensor(input_ids, dtype = torch.int32),
                            "attention_mask": torch.tensor(attention_mask, dtype = torch.int32),
                            "index": i,
                            "seq_index": j,
                            "masked_token": masked_token,
                            "length": 2
                        }
                    )
            
            for j, ids in enumerate(input_ids):
                temp_ids = output['input_ids'].copy()
                if (ids in [tokenizer.cls_token_id, tokenizer.sep_token_id]):
                    continue
                masked_token = temp_ids[j]
                temp_ids[j] = tokenizer.mask_token_id
                # print(f'masked_id:{masked_token}')

                # print(f'after_mask:{temp_ids}')
                
                data_list.append(
                    {
                        "name": name,
                        "input_ids": torch.tensor(temp_ids, dtype = torch.int32),
                        "attention_mask": torch.tensor(attention_mask, dtype = torch.int32),
                        "index": i,
                        "seq_index": j,
                        "masked_token": masked_token,
                        "length": len(input_ids) - 2
                    }
                )
    return LM_Dataset(data_list)
    
def getRescoreDataset(data_json, dataset, tokenizer, topk = 50):
    data_list = list()

    for i, key in enumerate(tqdm(data_json.keys())):
        # if (dataset in ['aishell', 'aishell2', 'old_aishell']):
        #     input_ids, _, attention_mask = tokenizer(data['ref']).values()
        # elif (dataset in ['tedlium2', 'librispeech']):
        #     input_ids, attention_mask = tokenizer(data['ref']).values()
        # print(data_json[key].keys())

        # if (i >= 64): break
        for hyp, score, rescore, err in zip(data_json[key]['hyps'], data_json[key]['score'], data_json[key]['Rescore'], data_json[key]['err']):
            if (dataset in ['aishell', 'aishell2', 'old_aishell']):
                input_ids, _, attention_mask = tokenizer(hyp).values()
            elif (dataset in ['tedlium2', 'librispeech']):
                input_ids, attention_mask = tokenizer(hyp).values()
            
            data_list.append(
                {
                    'name': key,
                    "input_ids":input_ids,
                    "attention_mask": attention_mask,
                    "score": score,
                    "mlm_score": rescore,
                    "err": err,
                    "wer": err['err']
                }
            )
    
    return LM_Dataset(data_list)

def getRecogDataset(data_json, dataset, tokenizer, topk = 50):
    data_list = list()

    for i, key in enumerate(tqdm(data_json.keys())):
        # if (dataset in ['aishell', 'aishell2', 'old_aishell']):
        #     input_ids, _, attention_mask = tokenizer(data['ref']).values()
        # elif (dataset in ['tedlium2', 'librispeech']):
        #     input_ids, attention_mask = tokenizer(data['ref']).values()
        # print(data_json[key].keys())
        for i, (hyp, score, rescore, err) in enumerate(zip(data_json[key]['hyps'], data_json[key]['score'],data_json[key]['Rescore'], data_json[key]['err'])):
            if (dataset in ['aishell', 'aishell2', 'old_aishell']):
                input_ids, _, attention_mask = tokenizer(hyp).values()
            elif (dataset in ['tedlium2', 'librispeech']):
                input_ids, attention_mask = tokenizer(hyp).values()
            
            data_list.append(
                {
                    "name": key,
                    "input_ids":input_ids,
                    "attention_mask": attention_mask,
                    "score": score,
                    "mlm_score": rescore,
                    "wer": err['err'],
                    'index': i
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

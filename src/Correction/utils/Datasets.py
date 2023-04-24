import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.nBestAligner.nBestAlign import align, alignNbest
from jiwer import wer

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
            if (not for_train):
                topk = 1
            for i, data in enumerate(tqdm(data_json, ncols = 80)):
                label = tokenizer(data['ref'])["input_ids"]
                ref = data['ref']

                for hyp in data['hyps'][:topk]:
                    if (wer(ref, hyp) > 0.3): continue
                    
                    output = tokenizer(hyp)
                    if ('token_type_ids' in output.keys()):
                        input_ids, _ , attention_mask = output.values()
                    else:
                        input_ids, attention_mask = output.values()
                    if (for_train):
                        data_list.append(
                            {
                                "input_ids":input_ids,
                                "attention_mask": attention_mask,
                                "labels": label,
                            }
                        )
                    else:
                        data_list.append(
                            {
                                "input_ids":input_ids,
                                "attention_mask": attention_mask,
                                "labels": label,
                                "top_hyp": data['hyps'][0]
                            }
                    )
                # if (i > 16): break
                    

        elif (data_type == 'concat'):

            if (sep_token == '[SEP]'):
                sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token

            for data in tqdm(data_json,ncols = 80):

                label = tokenizer(data['ref'])["input_ids"]
                concat_str = str()
                nbest = len(data['hyps']) if (len(data['hyps']) < topk) else topk
                for i in range(nbest):
                    # print(f"hyp - {i}:{data['hyps'][i]}")
                    if (i == 0):
                        concat_str = f"{data['hyps'][i]} {sep_token}"
                    elif (i == topk - 1):
                        concat_str = f"{concat_str} {data['hyps'][i]} {tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token}"
                    else:
                        concat_str = f"{concat_str} {data['hyps'][i]} {sep_token}"
                output = tokenizer(concat_str)

                # print(f'concat_str:{concat_str}')
                # print(f"concat_ids:{output['input_ids']}")

                if ('token_type_ids' in output.keys() ):
                    input_ids, _ , attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
                
                data_list.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": label,
                        "top_hyp": data['hyps'][0]
                    }
                )
            
        elif (data_type == 'align'):
            """
            Since the data file here is tghe string after align, we should get top hypothesis from "top_hyp" key made by gen_align.py
            """
            bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
            eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

            for i, data in enumerate(tqdm(data_json, ncols = 80)):
                input_ids = []
                input_ids.append([bos_token_id for _ in range(topk)])
                for token in data['hyps']:
                    token_ids = tokenizer.convert_tokens_to_ids(token)
                    convert_ids = [token for token in token_ids] # [token for ids in token_ids for token in ids]
                    assert(len(convert_ids) == topk), f"token:{token}, token_ids:{token_ids}, convert_ids:{convert_ids}"
                    input_ids.append(convert_ids)
                
                input_ids.append([eos_token_id for _ in range(topk)])

                # print(input_ids)
                
                label = tokenizer(data['ref'])["input_ids"]
                top_hyp = data['top_hyp']
                data_list.append(
                    {
                        "input_ids": input_ids,
                        "labels": label,
                        "top_hyp": top_hyp
                    }
                )
                
        return correctTrainerDataset(data_list)

    else:
        if (data_type == 'single'):
            for data in tqdm(data_json, ncols = 80):
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
                        "labels": label,
                        "top_hyp": data['hyps'][0]
                    }
                )

        elif (data_type == 'concat'):
            if (sep_token == '[SEP]'):
                sep_token = tokenizer.sep_token

            for data in tqdm(data_json, ncols = 80):
                name = data['name']
                concat_str = str()
                for i in range(topk):
                    if (i == 0):
                        concat_str = f"{data['hyps'][i]} {sep_token}"

                    elif (i == topk - 1):
                        concat_str = f"{concat_str} {data['hyps'][i]}"
                    else:
                        concat_str = f"{concat_str} {data['hyps'][i]} {sep_token}"
                
                output = tokenizer(concat_str)

                if ('token_type_ids' in output.keys()):
                    input_ids, _, attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
                
                label = tokenizer(data['ref'])["input_ids"]
                top_hyp = data['top_hyp']
                
                data_list.append(
                    {
                        "name": name,
                        "input_ids":input_ids,
                        "attention_mask": attention_mask,
                        "labels": label,
                        "top_hyp": data['hyps'][0]
                    }
                )
                    
        return correctTrainerDataset(data_list)
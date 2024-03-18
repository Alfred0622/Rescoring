import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.nBestAligner.nBestAlign import align, alignNbest
from jiwer import wer
from pyknp import Juman


jumanpp = Juman()

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
    
def change_unicode(token_list):
    for i, token in  enumerate(token_list):
        if (65281 <= ord(token) <= 65374 ): # if 全形
            token_list[i] = chr(ord(token) - 65248)
    
    return token_list

def preprocess_string(string, dataset):
    if (dataset in ['csj']):
        pass
        # string = change_unicode(string)
        # string = "".join(string)
        # print(f"string before segment:{string}")
        # string = jumanpp.analysis(string)
        # string = " ".join([mrph.genkei for mrph in string.mrph_list()])
        # print(f"string after segment:{string}")
    else:
        string = string.replace("<eos>", "").strip().split()
        string = [token for token in string]
        string = " ".join(string)
    
    return string

def get_dataset(data_json, dataset ,tokenizer, topk, sep_token = '[SEP]', data_type = 'single', for_train = True,fetch_num = -1):
    """
    data_type: str. can be "single" or "concat"
    """
    assert(isinstance(topk, int))
    if (sep_token == '[PAD]'):
        sep_token = tokenizer.pad_token
    elif (sep_token == '[SEP]'):
        sep_token = tokenizer.sep_token if (tokenizer.sep_token is not None) else tokenizer.eos_token
    print(f'sep_token:{sep_token}')

    if (topk < 1):
        topk = 1
    data_list = list()

    if (for_train):
        if (data_type == 'single'):
            if (not for_train):
                topk = 1
            for i, data in enumerate(tqdm(data_json, ncols = 80)):
                ref = preprocess_string(data['ref'], dataset)
                temp_ref = "".join(data['ref'].split())
                temp_ref = " ".join([t for t in temp_ref])
                label = tokenizer(ref)["input_ids"]

                for hyp in data['hyps'][:topk]:
                    temp_hyp = hyp.replace('<eos>', "").strip()
                    temp_hyp = "".join(temp_hyp.split())
                    temp_hyp = " ".join([t for t in temp_hyp])
                    if (wer([temp_ref], [temp_hyp]) > 0.3): continue

                    hyp = preprocess_string(hyp, dataset)
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
                                "top_hyp": data['hyps'][0].replace("<eos>", "").strip(),
                            }
                    )
                # if (i > 16): break
                    

        elif (data_type == 'concat'):

            if (sep_token == '[SEP]'):
                sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token

            for j, data in enumerate(tqdm(data_json,ncols = 80)):
                ref = preprocess_string(data['ref'], dataset)
                

                label = tokenizer(ref, max_length = 512,  truncation=True)["input_ids"]
                # if (len(label) < 3):
                #     print(f'ref:{ref}')
                #     print(f'label:{label}')
                concat_str = str()
                nbest = len(data['hyps']) if (len(data['hyps']) < topk) else topk
                for i in range(nbest):
                    hyp = preprocess_string(data['hyps'][i], dataset)
                    
                    if (i == 0):
                        concat_str = f"{hyp} {sep_token}" #  "Empty sequence" ->  {hyp} [SEP]
                    elif (i == topk - 1):
                        concat_str = f"{concat_str} {hyp}" # {hyp} [SEP] {hyp} [SEP] -> {hyp} [SEP] {hyp} [SEP] {last_hyp}
                    else:
                        concat_str = f"{concat_str} {hyp} {sep_token}" # {hyp} [SEP] -> {hyp} [SEP] {hyp} [SEP]
                
                output = tokenizer(concat_str, max_length = 512, truncation = True)

                if ('token_type_ids' in output.keys() ):
                    input_ids, _ , attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
                
                data_list.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": label,
                        "ref_text": ref,
                    }
                )
                if (fetch_num > 0 and i > fetch_num):
                    break
            
        elif (data_type == 'align'):
            """
            Since the data file here is tghe string after align, we should get top hypothesis from "top_hyp" key made by gen_align.py
            """
            
            bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
            eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

            if (sep_token == '[PAD]'):
                sep_token = tokenizer.pad_token
            elif (sep_token == '[SEP]'):
                sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
            elif (sep_token == '[UNK]'):
                sep_token = tokenizer.unk_token
            
            
            for i, data in enumerate(tqdm(data_json, ncols = 80)):
                input_ids = []
                hyps_text = []
                for hyp in data['hyps'][:topk]:
                    hyp = preprocess_string(hyp, dataset)
                    hyps_text.append(hyp)
                    token_ids = tokenizer(hyp)['input_ids'][:-1]
                    input_ids.append(token_ids)
                align_hyp_ids = align(input_ids, nBest = topk, placeholder = tokenizer.convert_tokens_to_ids(sep_token))

                if (len(align_hyp_ids) < topk - 1):
                    print(f'len:{len(align_hyp_ids)} , topk:{topk}')
                    print(f'align_hyp_ids:{align_hyp_ids}')
                    print(f'input_ids:{input_ids}')
                
                input_ids = alignNbest(align_hyp_ids, placeholder = tokenizer.convert_tokens_to_ids(sep_token))
 
                input_ids.append([eos_token_id for _ in range(topk)])

                # print(f'input_ids:{input_ids}')
                

                ref = preprocess_string(data['ref'], dataset)
                label = tokenizer(ref)["input_ids"]
                top_hyp = preprocess_string(data['hyps'][0], dataset)
                # print(f'label:{label}')
                data_list.append(
                    {
                        "name": data['name'],
                        "hyps_text": hyps_text,
                        "input_ids": input_ids,
                        "labels": label,
                        "top_hyp": top_hyp,
                        "ref_text": ref
                    }
                )

                if (fetch_num > 0 and i > fetch_num):
                    break
        data_list = sorted(data_list, key = lambda x: len(x['input_ids']))
                
        return correctTrainerDataset(data_list)

    else:
        if (data_type == 'single'):
            for data in tqdm(data_json, ncols = 80):
                name = data['name']
                hyp = preprocess_string(data['hyps'][0], dataset)
                output = tokenizer(hyp)
                if ('token_type_ids' in output.keys()):
                    input_ids, _ , attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()

                ref = preprocess_string(data['ref'], dataset)
                label = tokenizer(ref)["input_ids"]

                data_list.append(
                    {
                        "name": name,
                        "input_ids":input_ids,
                        "attention_mask": attention_mask,
                        "labels": label,
                        "top_hyp": data['hyps'][0].replace("<eos>", "").strip(),
                        "ref_text": data['ref'].replace("<eos>", "").strip()
                    }
                )

        elif (data_type == 'concat'):
            if (sep_token == '[SEP]'):
                sep_token = tokenizer.sep_token

            for j, data in enumerate(tqdm(data_json, ncols = 80)):

                name = data['name']
                concat_str = str()
                for i in range(topk):
                    hyp = preprocess_string(data['hyps'][i], dataset)
                    
                    if (i == 0):
                        concat_str = f"{hyp} {sep_token}"
                    elif (i == topk - 1):
                        concat_str = f"{concat_str} {hyp}"
                    else:
                        concat_str = f"{concat_str} {hyp} {sep_token}"
                
                ref = preprocess_string(data['ref'], dataset)
                top_hyp = preprocess_string(data['hyps'][0], dataset)
                
                output = tokenizer(concat_str, max_length = 512, truncation = True)

                if ('token_type_ids' in output.keys()):
                    input_ids, _, attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
                
                label_tokens = tokenizer(ref)["input_ids"]
                top_hyp_tokens = tokenizer(top_hyp)["input_ids"]
                
                data_list.append(
                    {
                        "name": name,
                        "input_ids":input_ids,
                        "attention_mask": attention_mask,
                        "labels": label_tokens,
                        "ref_text": data['ref'].replace("<eos>", "").strip(),
                        "top_hyp": data['hyps'][0].replace("<eos>", "").strip(),
                        "top_hyp_token": top_hyp_tokens
                    }
                )
        
        data_list = sorted(data_list, key = lambda x: len(x['input_ids']))
                    
        return correctTrainerDataset(data_list)
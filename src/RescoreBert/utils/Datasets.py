import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn import Softmax


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
    max_len = 0.0

    if (for_train):
        if (lm == "MLM"):
            for data in tqdm(data_json, ncols = 100):

                if (dataset in ['csj']):
                    if (jp_split): 
                        ref = data['ref']
                    else:
                        ref = "".join(data['ref'].split())
                else:
                    ref = data['ref']
                output = tokenizer(ref)

                if ('token_type_ids' in output.keys()):
                    input_ids, _, attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
                
                if (len(input_ids) > max_len):
                    max_len = len(input_ids)

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
            bos_token = tokenizer.cls_token if tokenizer.bos_token is None else tokenizer.bos_token
            eos_token = tokenizer.sep_token if tokenizer.eos_token is None else tokenizer.eos_token
            for i, data in enumerate(tqdm(data_json, ncols = 100)):
                
                if (dataset in ['csj']):
                    if (jp_split): 
                        ref = data['ref']
                    else:
                        ref = "".join(data['ref'].split())

                elif (dataset in ['tedlium2', 'tedlium2_conformer', 'librispeech']):
                    ref = data['ref'] + "."
                else:
                    ref = data['ref']

                if (dataset in ['aishell', 'aishell2']):
                    pass
                else:
                    ref = f'{bos_token} {ref} {eos_token}'

                output = tokenizer(ref)

                if ('token_type_ids' in output.keys()):
                    input_ids, _ , attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()

                if (len(input_ids) > max_len):
                    max_len = len(input_ids)
                if (len (input_ids) <= 1): continue
                input_ids = torch.tensor(input_ids, dtype = torch.int32)
                data_list.append(
                    {
                        "input_ids": input_ids,
                    }
                )
            print(f'# num of Dataset:{len(data_list)}')

        print(f'max_len:{max_len}')


        return LM_Dataset(data_list)

    else:
        for k, data in enumerate(tqdm(data_json, ncols = 100)):
            if (topk > len(data["hyps"])):
                nbest = len(data["hyps"])
            else: nbest = topk

            name = data['name']
            for i, hyp in enumerate(data['hyps'][:nbest]):
                if ("<eos>" in hyp):
                    hyp = hyp[:-5]
                
                if (dataset in ['csj']):
                    if (jp_split): 
                        hyp = hyp
                    else:
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
            
            # if (k > 50):
            #     break
        return LM_Dataset(data_list)

def get_mlm_dataset(data_json, tokenizer,  dataset, topk = 50, jp_split = True):

    bos_id = tokenizer.cls_token_id if tokenizer.bos_token_id is None else tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.eos_token_id is None else tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    data_list = list()

    assert (bos_id is not None and eos_id is not None and mask_id is not None), f"{bos_id}, {eos_id}, {mask_id}"

    for k, data in enumerate(tqdm(data_json, ncols = 100)):
        if (topk > len(data["hyps"])):
            nbest = len(data["hyps"])
        else: nbest = topk
        name = data['name']

        for i, hyp in enumerate(data['hyps'][:nbest]):
            if ("<eos>" in hyp):
                hyp = hyp[:hyp.find('<eos>')]
            if (dataset in ['csj']):
                if (jp_split): 
                    hyp = hyp
                else:
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

        # if (k > 120):
        #     break    
    return LM_Dataset(data_list)
    
def getRescoreDataset(data_json, dataset, tokenizer, mode,topk = 50):
    data_list = list()
    assert (mode in ['MD', 'MWER', 'MWED']), "Modes should be MD, MWER or MWED"
     
    if (mode == 'MD'):
        if (isinstance(data_json, dict)):
            for i, key in enumerate(tqdm(data_json.keys(), ncols = 100)):
                wers = []

                for err in data_json[key]['err']:
                    wers.append(err['err'])
                
                wers = torch.tensor(wers, dtype = torch.float32)

                avg_err = torch.mean(wers).item()
                for hyp, score, rescore, err in zip(data_json[key]['hyps'], data_json[key]['score'] ,data_json[key]['rescore'], data_json[key]['err']):
                    output = tokenizer(hyp)
                    if ('token_type_ids' in output.keys()):
                        input_ids, _, attention_mask = tokenizer(hyp).values()
                    else:
                        input_ids, attention_mask = tokenizer(hyp).values()
                
                    
                    data_list.append(
                        {
                            'name': key,
                            "input_ids":input_ids,
                            "attention_mask": attention_mask,
                            "score": score,
                            "mlm_score": rescore,
                            "err": err,
                            "wer": err['err'],
                            "avg_err": avg_err
                        }
                    )
                # if (i > 550):
                #     break
            
        elif (isinstance(data_json, list)):
            for i, data in enumerate(tqdm(data_json, ncols = 100)):
                wers = []

                for err in data['err']:
                    wers.append(err['err'])
                
                wers = torch.tensor(wers, dtype = torch.float32)
                avg_err = wers.mean().item()
                
                for hyp, score, rescore, err in zip(data['hyps'], data['score'] ,data['rescore'], data['err']):
                    output = tokenizer(hyp)
                    if ('token_type_ids' in output.keys()):
                        input_ids, _, attention_mask = tokenizer(hyp).values()
                    else:
                        input_ids, attention_mask = tokenizer(hyp).values()
                
                    
                    data_list.append(
                        {
                            'name': data['name'],
                            "input_ids":input_ids,
                            "attention_mask": attention_mask,
                            "score": score,
                            "mlm_score": rescore,
                            "err": err,
                            "wer": err['err'],
                            "avg_err": avg_err
                        }
                    )
                # if (i > 550):
                #     break
            
    elif (mode in ['MWER', 'MWED']):
        if (isinstance(data_json, dict)):
            for i, key in enumerate(tqdm(data_json.keys(), ncols = 100)):
                wers = []
                for err in data_json[key]['err']:
                    wers.append(err['err'])
                
                wers_tensor = torch.tensor(wers, dtype = torch.float32)
                avg_err = torch.mean(wers_tensor).item()

                scores = torch.tensor(data_json[key]['score'])

                input_ids = []
                attention_masks = []
                avg_errs = []

                for hyp in data_json[key]['hyps']:
                    output = tokenizer(hyp)
                    input_ids.append(output['input_ids'])
                    attention_masks.append(output['attention_mask'])
                    avg_errs.append(avg_err)
                
                nbest = len(data_json[key]['hyps'])

                data_list.append(
                    {
                        "hyps":data_json[key]['hyps'], 
                        "name":key,
                        "input_ids": input_ids,
                        "attention_mask": attention_masks,
                        "score": data_json[key]['score'],
                        "rescore": data_json[key]['rescore'],
                        'errs':data_json[key]['err'],
                        "wer": wers,
                        "avg_err": avg_errs,
                        "nbest": nbest
                    }
                )
                # if (i > 32):
                #     break
            
        elif (isinstance(data_json, list)):
            for i, data in enumerate(tqdm(data_json, ncols = 100)):
                wers = []
                for err in data['err']:
                    wers.append(err['err'])
                
                wers_tensor = torch.tensor(wers, dtype = torch.float32)
                avg_err = torch.mean(wers_tensor).item()

                scores = torch.tensor(data['score'])

                input_ids = []
                attention_masks = []
                avg_errs = []
                for hyp in data['hyps']:
                    output = tokenizer(hyp)
                    input_ids.append(output['input_ids'])
                    attention_masks.append(output['attention_mask'])
                    avg_errs.append(avg_err)
                
                nbest = len(data['hyps'])

                data_list.append(
                    {
                        "hyps":data['hyps'], 
                        "name":data['name'],
                        "input_ids": input_ids,
                        "attention_mask": attention_masks,
                        "score": data['score'],
                        "rescore": data['rescore'],
                        'errs':data['err'],
                        "wer": wers,
                        "avg_err": avg_errs,
                        "nbest": nbest
                    }
                )
                # if (i > 32):
                #     break

    return LM_Dataset(data_list)

def getRecogDataset(data_json, dataset, tokenizer, topk = 50):
    data_list = list()

    if (isinstance(data_json, dict)):
        for i, key in enumerate(tqdm(data_json.keys(), ncols = 100)):
            # if (dataset in ['aishell', 'aishell2', 'old_aishell']):
            #     input_ids, _, attention_mask = tokenizer(data['ref']).values()
            # elif (dataset in ['tedlium2', 'librispeech']):
            #     input_ids, attention_mask = tokenizer(data['ref']).values()
            for j, (hyp, score, rescore, err) in enumerate(zip(data_json[key]['hyps'], data_json[key]['score'],data_json[key]['rescore'], data_json[key]['err'])):
                output = tokenizer(hyp)
                
                if ('token_type_ids' in output.keys()):
                    input_ids, _, attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
                
                data_list.append(
                    {
                        "name": key,
                        "input_ids":input_ids,
                        "attention_mask": attention_mask,
                        "score": score,
                        "mlm_score": rescore,
                        "wer": err['err'],
                        'index': j,
                        "top_hyp": data_json[key]['hyps'][0]
                    }
                )
    elif (isinstance(data_json, list)):
        for i, data in enumerate(tqdm(data_json, ncols = 100)):
            # if (dataset in ['aishell', 'aishell2', 'old_aishell']):
            #     input_ids, _, attention_mask = tokenizer(data['ref']).values()
            # elif (dataset in ['tedlium2', 'librispeech']):
            #     input_ids, attention_mask = tokenizer(data['ref']).values()
            # print(data_json[key].keys())
            for j, (hyp, score, rescore, err) in enumerate(zip(data['hyps'], data['score'],data['rescore'], data['err'])):
                output = tokenizer(hyp)
                
                if ('token_type_ids' in output.keys()):
                    input_ids, _, attention_mask = output.values()
                else:
                    input_ids, attention_mask = output.values()
                
                data_list.append(
                    {
                        "name": data['name'],
                        "input_ids":input_ids,
                        "attention_mask": attention_mask,
                        "score": score,
                        "mlm_score": rescore,
                        "wer": err['err'],
                        'index': j,
                        "top_hyp": data['hyps'][0]
                    }
                )
        
    
    return LM_Dataset(data_list)

def prepare_myDataset(data_json, bert_tokenizer, gpt_tokenizer, topk = 50):
    data_list = list()
    print(f'prepare_MyDataset')
    if (isinstance(data_json, list)):
        for i, data in enumerate(tqdm(data_json, ncols = 100)):
            assert(isinstance(data['hyps'], list)), f"Hyps is not list:{data['hyps']}"
            assert(isinstance(data['am_score'], list)), f"{data['name']} -- Am_score is not list:{data['am_score']}, hyps = {data['hyps']}, ctc_score = {data['ctc_score']}"
            assert(isinstance(data['ctc_score'], list)), f"CTC_score is not list:{data['ctc_score']}"
            assert(isinstance(data['err'], list)), f"WER is not list:{data['err']}"


            for j, (hyp, am_score, ctc_score, err) in enumerate(zip(data['hyps'], data['am_score'],data['ctc_score'], data['err'])):
                bert_output = bert_tokenizer(hyp)

                gpt_output = gpt_tokenizer(hyp)

                if ('token_type_ids' in bert_output.keys()):
                    input_ids, _ , attention_mask = bert_output.values()
                
                else:
                    input_ids, attention_mask = bert_output.values()

                
                if ('token_type_ids' in gpt_output.keys()):
                    gpt_input_ids, _, gpt_attention_mask = gpt_output.values()
                
                else:
                    gpt_input_ids, gpt_attention_mask = gpt_output.values()

                
                
                data_list.append(
                    {
                        'name': data['name'],
                        'bert_input_ids': input_ids,
                        "bert_mask": attention_mask,
                        "gpt_input_ids": gpt_input_ids,
                        "gpt_mask": gpt_attention_mask,
                        "am_score": am_score,
                        "ctc_score": ctc_score,
                        "wer": err['err'],
                        'index': j
                    }
                )
            # if (i > 256):
            #     break        

    elif (isinstance(data_json, dict)):
        for i, key in enumerate(tqdm(data_json.keys(), ncols = 100)):
            for j, (hyp, score, rescore, err) in enumerate(zip(data[key]['hyps'], data[key]['score'],data[key]['rescore'], data[key]['err'])):
                bert_output = bert_tokenizer(hyp)

                gpt_output = gpt_tokenizer()

                if ('token_type_ids' in bert_output.keys()):
                    input_ids, _ , attention_mask = bert_output.values()
                
                else:
                    input_ids, attention_mask = bert_output.values()

                
                if ('token_type_ids' in gpt_output.keys()):
                    gpt_input_ids, _, gpt_attention_mask = gpt_output.values()
                
                else:
                    gpt_input_ids, gpt_attention_mask = gpt_output.values()

                
                
                data_list.append(
                    {
                        'name': data['name'],
                        'bert_input_ids': input_ids,
                        "bert_mask": attention_mask,
                        "gpt_input": gpt_input_ids,
                        "gpt_mask": gpt_attention_mask,
                        "score": score,
                        "mlm_score": rescore,
                        "wer": err['err'],
                        'index': j

                    }
                )
            # if (i > 256):
            #     break
    
    return LM_Dataset(data_list)


def prepare_myRecogDataset(data_json, bert_tokenizer, topk = 50):
    data_list = list()
    print(f'prepare_MyDataset')
    if (isinstance(data_json, list)):
        for i, data in enumerate(tqdm(data_json, ncols = 100)):
            assert(isinstance(data['hyps'], list)), f"Hyps is not list:{data['hyps']}"
            assert(isinstance(data['am_score'], list)), f"{data['name']} -- Am_score is not list:{data['am_score']}, hyps = {data['hyps']}, ctc_score = {data['ctc_score']}"
            assert(isinstance(data['ctc_score'], list)), f"CTC_score is not list:{data['ctc_score']}"
            assert(isinstance(data['err'], list)), f"WER is not list:{data['err']}"


            for j, (hyp, am_score, ctc_score, err) in enumerate(zip(data['hyps'], data['am_score'],data['ctc_score'], data['err'])):
                bert_output = bert_tokenizer(hyp)


                if ('token_type_ids' in bert_output.keys()):
                    input_ids, _ , attention_mask = bert_output.values()
                
                else:
                    input_ids, attention_mask = bert_output.values()
                
                data_list.append(
                    {
                        'name': data['name'],
                        'bert_input_ids': input_ids,
                        "bert_mask": attention_mask,
                        "am_score": am_score,
                        "ctc_score": ctc_score,
                        "wer": err['err'],
                        'index': j
                    }
                )
            # if (i > 256):
            #     break        

    elif (isinstance(data_json, dict)):
        for i, key in enumerate(tqdm(data_json.keys(), ncols = 100)):
            for j, (hyp, score, rescore, err) in enumerate(zip(data[key]['hyps'], data[key]['score'],data[key]['rescore'], data[key]['err'])):
                bert_output = bert_tokenizer(hyp)


                if ('token_type_ids' in bert_output.keys()):
                    input_ids, _ , attention_mask = bert_output.values()
                
                else:
                    input_ids, attention_mask = bert_output.values()
                
                data_list.append(
                    {
                        'name': data['name'],
                        'bert_input_ids': input_ids,
                        "bert_mask": attention_mask,
                        "score": score,
                        "mlm_score": rescore,
                        "wer": err['err'],
                        'index': j
                    }
                )
            # if (i > 256):
            #     break
    
    return LM_Dataset(data_list)


def prepareListwiseDataset(data_json, tokenizer, topk = 50, sort_by_len = False):
    """
    The purpose of the function is to get the complete dataset. Includes:

    - utt name
    - hyps (tokenized)
    - refs
    - am_score
    - ctc_score
    - WER (Include correct, sub, ins, del error number)
    - WER only
    - average WER
    - len of hyps

    In this function, we will NOT split the N-best List
    """

    data_list = list()
    if (isinstance(data_json, dict)):
        for i, key in enumerate(tqdm(data_json.keys(), ncols = 100)):
            wers = []
            for err in data_json[key]['err']:
                wers.append(err['err'])
            
            wers_tensor = torch.as_tensor(wers, dtype = torch.float32)
            avg_err = torch.mean(wers_tensor).item()

            scores = torch.as_tensor(data_json[key]['score'], dtype = torch.float32)
            am_scores = torch.as_tensor(data_json[key]['am_score'], dtype = torch.float32)
            ctc_scores = torch.as_tensor(data_json[key]['ctc_score'], dtype = torch.float32)
            # lm_scores = torch.as_tensor(data_json[key]['lm_score'])

            input_ids = []
            attention_masks = []
            avg_errs = []

            min_len = 10000
            max_len = -1

            for hyp in data_json[key]['hyps']:
                output = tokenizer(hyp)
                input_ids.append(output['input_ids'])
                attention_masks.append(output['attention_mask'])
                avg_errs.append(avg_err)

                if (len(hyp) > max_len):
                    max_len = len(hyp)
                if (len(hyp) < min_len):
                    min_len = len(hyp)
            
            nbest = len(data_json[key]['hyps'])


            data_list.append(
                {
                    "name":key,
                    "hyps":data_json[key]['hyps'], 
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                    "score": scores,
                    "am_score": am_scores,
                    "ctc_score": ctc_scores,
                    'errs':data_json[key]['err'],
                    "wer": wers_tensor,
                    "avg_err": avg_errs,
                    "nbest": nbest,
                    "max_len": max_len,
                    "min_len": min_len
                }
            )
            # if (i > 18):
            #     break
        
    elif (isinstance(data_json, list)):
        for i, data in enumerate(tqdm(data_json, ncols = 100)):
            wers = []
            for err in data['err']:
                wers.append(err['err'])
            
            wers_tensor = torch.tensor(wers, dtype = torch.float32)
            avg_err = torch.mean(wers_tensor).item()

            scores = torch.as_tensor(data['score'], dtype = torch.float32)
            am_scores = torch.as_tensor(data['am_score'], dtype = torch.float32)
            ctc_scores = torch.as_tensor(data['ctc_score'], dtype = torch.float32)

            input_ids = []
            attention_masks = []
            avg_errs = []

            min_len = 10000
            max_len = -1
            
            for hyp in data['hyps']:
                output = tokenizer(hyp)
                input_ids.append(output['input_ids'])
                attention_masks.append(output['attention_mask'])
                avg_errs.append(avg_err)

                if (len(output['input_ids']) > max_len):
                    max_len = len(output['input_ids'])
                if (len(output['input_ids']) < min_len):
                    min_len = len(output['input_ids'])
            
            nbest = len(data['hyps'])
            # print(f'nbest:{nbest}')



            data_list.append(
                {
                    "hyps":data['hyps'], 
                    "name":data['name'],
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                    "score": scores,
                    "am_score": am_scores,
                    "ctc_score": ctc_scores,
                    'errs':data['err'],
                    "wer": wers_tensor,
                    "avg_err": avg_errs,
                    "nbest": nbest,
                    "max_len": max_len,
                    "min_len": min_len
                }
            )
            # if (i > 2560):
            #     break
    
    if (sort_by_len):
        data_list = sorted(data_list, key = lambda x: x['max_len'])
    return LM_Dataset(data_list)
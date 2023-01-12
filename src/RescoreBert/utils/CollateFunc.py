import torch
import logging
from torch.nn.utils.rnn import pad_sequence

def recogBatch(batch):
    names = []
    indexs = []
    input_ids = []
    attention_mask = []

    for sample in batch:
        indexs.append(sample['index'])
        names.append(sample['name'])
        input_ids.append(torch.tensor(sample['input_ids'], dtype = torch.long))
        # token_type_ids.append(torch.tensor(sample['token_type_ids'], dtype = torch.long))
        attention_mask.append(torch.tensor(sample['attention_mask'], dtype = torch.long))
    
    input_ids = pad_sequence(input_ids, batch_first = True)
    # token_type_ids = pad_sequence(token_type_ids, batch_first = True)
    attention_mask = pad_sequence(attention_mask, batch_first = True)

    return(
        {
            "input_ids": input_ids,
            # "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "name": names,
            "index": indexs
        }
    )

def RescoreBertBatch(batch):
    input_ids = []
    attention_masks = []
    scores = []
    labels = []
    errs = []
    wers = []
    for sample in batch:
        input_ids.append(torch.tensor(sample['input_ids'], dtype = torch.int64))
        attention_masks.append(torch.tensor(sample['attention_mask'], dtype = torch.int64))
        scores.append(sample['score'])
        labels.append(sample['mlm_score'])
        errs.append(sample['err'])
        wers.append(sample['wer'])
    
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    scores = torch.tensor(scores, dtype = torch.float64)
    labels = torch.tensor(labels, dtype = torch.float64)
    wers = torch.tensor(wers, dtype = torch.float64)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "score": scores,
        "labels": labels,
        "err": errs,
        "wer": wers
    }

def RescoreBertRecogBatch(batch):
    names = []
    input_ids = []
    attention_masks = []
    labels = []
    wers = []
    indexes = []
    for sample in batch:
        names.append(sample['name'])
        input_ids.append(torch.tensor(sample['input_ids'], dtype = torch.int64))
        attention_masks.append(torch.tensor(sample['attention_mask'], dtype = torch.int64))
        labels.append(sample['score'])
        wers.append(sample['wer'])
        indexes.append(sample['index'])
    
    assert(len(names) == len(input_ids)), f"input_ids length {len(input_ids)} != name length {len(names)}"
    assert(len(names) == len(attention_masks)), f"input_ids length {len(names)} != name length {len(attention_masks)}"
    assert(len(names) == len(labels)), f"input_ids length {len(names)} != name length {len(labels)}"
    assert(len(names) == len(wers)), f"input_ids length {len(names)} != name length {len(wers)}"
    assert(len(names) == len(labels)), f"input_ids length {len(names)} != name length {len(labels)}"

    
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    labels = torch.tensor(labels, dtype = torch.float64)
    wers = torch.tensor(wers, dtype = torch.float64)

    return {
        "name": names,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "wer": wers,
        "index": indexes
    }


def recogMLMBatch(batch):
    names = []
    input_ids = []
    attention_mask = []
    masked_tokens = []
    nBest_index = []
    seq_index = []
    lengths = []

    for sample in batch:
        names.append(sample['name'])
        input_ids.append(sample['input_ids'])
        attention_mask.append(sample['attention_mask'])

        masked_tokens.append(sample['masked_token'])
        nBest_index.append(sample['index'])
        seq_index.append(sample['seq_index'])

        lengths.append(sample['length'])
        # bos_id = sample['bos_id']
        # eos_id = sample['eos_id']
        # mask_id = sample['mask_id']

        # if (len(sample['input_ids']) == 2):

        #     names.append(sample['name'])
        #     input_ids.append(sample['input_ids'])
        #     attention_mask.append(sample['attention_mask'])
        #     seq_index.append(-1)
        #     masked_tokens.append(-1)
        #     nBest_index.append(sample['nbest'])

            # if (len(batch) == 1):
            #     return {
            #         "name": [sample['name']],
            #         "input_ids": None,
            #         "attention_mask": None, 
            #         "seq_index": None,
            #         "masked_token": None,
            #         "nBest_index": None,
            #         "penalty_score": -10000
            #     }

            # continue

    #     else:
    #         for i, token in enumerate(sample['input_ids']):
    #             temp_ids = sample['input_ids'].clone()
    #             if (token in [bos_id, eos_id]):
    #                 continue
    #             names.append(sample['name'])
    #             temp_ids[i] = mask_id

    #             input_ids.append(temp_ids)
    #             attention_mask.append(sample['attention_mask'])
    #             seq_index.append(i) # append index of token
    #             masked_tokens.append(token.item()) # append masked token
    #             nBest_index.append(sample['nbest']) # append which nbest is
    
    data_num = len(names)

    assert(len(input_ids) == data_num), f'data_num: {data_num} != len(input_ids): {len(input_ids)}'
    assert(len(attention_mask) == data_num), f'data_num: {data_num} != len(input_ids): {len(attention_mask)}'
    assert(len(seq_index) == data_num), f'data_num: {data_num} != len(input_ids): {len(seq_index)}'
    assert(len(masked_tokens) == data_num), f'data_num: {data_num} != len(input_ids): {len(masked_tokens)}'
    assert(len(nBest_index) == data_num), f'data_num: {data_num} != len(input_ids): {len(nBest_index)}'

    # assert(len(input_ids) > 0), f'{input_ids}'
    
    input_ids = pad_sequence(input_ids, batch_first = True)
    attention_mask = pad_sequence(attention_mask, batch_first = True)

    return (
        {
            "name": names,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_index": seq_index,
            "masked_token": masked_tokens,
            "nBest_index": nBest_index,
            "length": lengths
        }
    )

def adaptionBatch(sample):
    tokens = [torch.tensor(s) for s in sample]

    tokens = pad_sequence(tokens, batch_first=True)

    # segs = pad_sequence(segs, batch_first=True)

    masks = torch.zeros(tokens.shape, dtype=torch.long)
    masks = masks.masked_fill(tokens != 0, 1)

    return tokens, masks

# pll scoring & recognizing
def pllScoringBatch(sample):
    name = [s[0] for s in sample]

    tokens = []
    for s in sample:
        tokens += s[1]

    texts = []
    for s in sample:
        texts += s[2]

    scores = []
    for s in sample:
        scores += s[3]

    ref = [s[4] for s in sample]

    cer = [s[5] for s in sample]

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    # for i, s in enumerate(segs):
    #     segs[i] = torch.tensor(s)

    tokens = pad_sequence(tokens, batch_first=True)

    masks = torch.zeros(tokens.shape, dtype=torch.long)
    masks = masks.masked_fill(tokens != 0, 1)

    return name[0], tokens, texts, masks, torch.tensor(scores), ref, cer

#  MD distillation
def rescoreBertBatch(sample):

    tokens = []
    texts = []

    for s in sample:
        tokens += s[0]

    texts = []
    for s in sample:
        texts += s[1]

    scores = []
    for s in sample:
        scores += s[2]

    cers = []
    for s in sample:
        cers += s[3]
    plls = []
    for s in sample:
        plls += s[4]

    assert len(plls) == len(tokens), f"illegal pll:{len(plls)} != {len(tokens)}"
    plls = torch.tensor(plls)

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    # for i, s in enumerate(segs):
    #     segs[i] = torch.tensor(s)

    tokens = pad_sequence(tokens, batch_first=True)

    # segs = pad_sequence(segs, batch_first=True)

    masks = torch.zeros(tokens.shape, dtype=torch.long)
    masks = masks.masked_fill(tokens != 0, 1)

    return tokens, texts, masks, torch.tensor(scores), torch.tensor(cers), plls

# RescoreBertRecog
def RescoreBertRecog(sample):
    # using with rescoreDataset
    # s[0] : name
    # s[1] : token
    # s[2] : text for hyp
    # s[3] : score
    # s[4] : ref
    # s[5] : err
    names = []
    tokens = []
    scores = []
    texts = []
    refs = []
    cers = []

    for s in sample:
        names += s[0]
        tokens += s[1]
        texts += s[2]
        scores += s[3]
        refs += s[4]
        cers += s[5]

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    # for i, s in enumerate(segs):
    #     segs[i] = torch.tensor(s)

    tokens = pad_sequence(tokens, batch_first=True)

    masks = torch.zeros(tokens.shape, dtype=torch.long)
    masks = masks.masked_fill(tokens != 0, 1)

    return names, tokens, masks, scores, texts, refs, cers
 
def lmBatch(sample):
    tokens = []
    labels = []
    cers = []
    scores = []

    for s in sample:
        cers += s[2]
        scores += s[3]
        for i, t in enumerate(s[0]):
            tokens.append(torch.tensor(t))
        labels.append(torch.tensor(s[1]))

    tokens = pad_sequence(tokens, batch_first = True)
    labels = pad_sequence(labels, batch_first = True)

    attention_masks = torch.zeros(tokens.shape)
    attention_masks[tokens != 0] = 1
        
    label_mask = torch.zeros(labels.shape)
    label_mask[labels != 0] = 1

    return (
        tokens, 
        attention_masks,
        labels,
        label_mask,
        torch.tensor(cers), 
        torch.tensor(scores)
    )

def lmRecogBatch(sample):
    tokens = []
    labels = []
    texts = []
    cers = []
    scores = []
    ref = []

    for s in sample:
        for i, t in enumerate(s[0]):
            tokens.append(torch.tensor(t))

        tokens = pad_sequence(tokens, batch_first = True)

        attention_masks = torch.zeros(tokens.shape)
        attention_masks[tokens != 0] = 1
        
        cers += s[2]
        scores += s[3]
        texts += s[4]
        ref += s[5]
        
    return (
        tokens, 
        attention_masks, 
        torch.tensor(cers), 
        torch.tensor(scores),
        texts,
        ref
    )

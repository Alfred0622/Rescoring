from re import L
import torch
import logging
from torch.nn.utils.rnn import pad_sequence

def bertCompareBatch(batch):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []
    for sample in batch:
        input_ids.append(torch.tensor(sample["input_ids"], dtype = torch.long))
        token_type_ids.append(torch.tensor(sample["token_type_ids"], dtype = torch.long))
        attention_mask.append(torch.tensor(sample["attention_mask"], dtype = torch.long))
        labels.append(sample['labels'])
    
    input_ids = pad_sequence(input_ids, batch_first = True)
    token_type_ids = pad_sequence(token_type_ids, batch_first = True, padding_value=1)
    attention_mask = pad_sequence(attention_mask, batch_first = True)
    labels = torch.tensor(labels, dtype = torch.float32)
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def recogBatch(batch):

    # for this function, recogBatch will be always 1
    name = []
    input_ids = []
    token_type_ids = []
    attention_mask = []
    pairs = []

    for sample in batch:
        name.append(sample['name'])
        input_ids.append(torch.tensor(sample['input_ids'], dtype=torch.int32))
        token_type_ids.append(torch.tensor(sample['token_type_ids'], dtype = torch.int32))
        attention_mask.append(torch.tensor(sample['attention_mask'], dtype = torch.int32))
        pairs.append(sample['pair'])
    
    assert(len(name) == len(input_ids)), f"input_ids length {len(input_ids)} != name length {len(name)}"
    assert(len(name) == len(attention_mask)), f"input_ids length {len(input_ids)} != name length {len(attention_mask)}"
    assert(len(name) == len(token_type_ids)), f"input_ids length {len(input_ids)} != name length {len(token_type_ids)}"
    assert(len(name) == len(pairs)), f"input_ids length {len(input_ids)} != name length {len(pairs)}"


    input_ids = pad_sequence(input_ids, batch_first = True)
    token_type_ids = pad_sequence(token_type_ids, batch_first = True, padding_value= 1)
    attention_mask = pad_sequence(attention_mask, batch_first = True)
    
    return {
        "name": name,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "pair": pairs,
    }

def recogWholeBatch(batch):
    name = []

    input_ids = [sample['input_ids'] for sample in batch]
    token_type_ids = [sample['token_type_ids'] for sample in batch]
    attention_mask = [sample['attention_mask'] for sample in batch]
    am_scores = [sample['am_score'] for sample in batch]
    ctc_scores = [sample['ctc_score'] for sample in batch]
    lm_scores = [sample['lm_score'] for sample in batch]
    scores = [sample['score'] for sample in batch]

    input_ids = pad_sequence(input_ids , batch_first = True)
    token_type_ids = pad_sequence(token_type_ids , batch_first = True, padding_value=1)
    attention_mask = pad_sequence(attention_mask, batch_first = True)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "am_score": am_scores,
        "ctc_score": ctc_scores,
        "lm_score": lm_scores
    }

def bertAlsemBatch(batch):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []

    am_scores = []
    ctc_scores = []
    lm_scores = []

    for sample in batch:
        input_ids.append(torch.tensor(sample["input_ids"], dtype = torch.long))
        token_type_ids.append(torch.tensor(sample["token_type_ids"], dtype = torch.long))
        attention_mask.append(torch.tensor(sample["attention_mask"], dtype = torch.long))
        labels.append(sample['labels'])

        am_scores.append(sample['am_score'])
        ctc_scores.append(sample['ctc_score'])
        lm_scores.append(sample['lm_score'])
    
    input_ids = pad_sequence(input_ids, batch_first = True)
    token_type_ids = pad_sequence(token_type_ids, batch_first = True, padding_value=1)
    attention_mask = pad_sequence(attention_mask, batch_first = True)
    labels = torch.tensor(labels, dtype = torch.float32)

    am_scores = torch.stack(am_scores)
    ctc_scores = torch.stack(ctc_scores)
    lm_scores = torch.stack(lm_scores)
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "am_score": am_scores,
        "ctc_score": ctc_scores,
        "lm_score": lm_scores,
    }

def recogAlsemBatch(batch):
    
    # for this function, recogBatch will be always 1
    name = []
    input_ids = []
    token_type_ids = []
    attention_mask = []
    am_scores = []
    ctc_scores = []
    lm_scores = []
    pairs = []

    for sample in batch:
        name.append(sample['name'])
        input_ids.append(torch.tensor(sample['input_ids'], dtype=torch.int32))
        token_type_ids.append(torch.tensor(sample['token_type_ids'], dtype = torch.int32))
        attention_mask.append(torch.tensor(sample['attention_mask'], dtype = torch.int32))
        pairs.append(sample['pair'])
        am_scores.append(torch.tensor(sample['am_score']))
        ctc_scores.append(torch.tensor(sample['ctc_score']))
        lm_scores.append(torch.tensor(sample['lm_score']))

    input_ids = pad_sequence(input_ids, batch_first = True)
    token_type_ids = pad_sequence(token_type_ids, batch_first = True, padding_value= 1)
    attention_mask = pad_sequence(attention_mask, batch_first = True)
    am_scores = torch.stack(am_scores)
    ctc_scores = torch.stack(ctc_scores)
    lm_scores = torch.stack(lm_scores)

    return {
        "name": name,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "pair": pairs,
        "am_score": am_scores,
        "ctc_score": ctc_scores,
        "lm_score": lm_scores
    }

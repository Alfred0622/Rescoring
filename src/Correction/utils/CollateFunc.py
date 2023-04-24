import torch
import logging
from torch.nn.utils.rnn import pad_sequence

def trainBatch(batch):
    input_ids = []
    attention_mask = []
    labels = []

    for sample in batch:
        input_ids.append(torch.tensor(sample["input_ids"], dtype = torch.long))
        attention_mask.append(torch.tensor(sample["attention_mask"], dtype = torch.long))
        labels.append(torch.tensor(sample["labels"], dtype = torch.long))
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value = 0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def recogBatch(batch):
    names = []
    input_ids = []
    attention_mask = []
    refs = []

    for sample in batch:
        names.append(sample["name"])
        input_ids.append(torch.tensor(sample["input_ids"], dtype = torch.long))
        attention_mask.append(torch.tensor(sample["attention_mask"], dtype = torch.long))
        refs.append(sample["labels"])
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value = 0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "name": names,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": refs
    }

def nBestAlignBatch(batch):
    input_ids = [torch.stack([torch.tensor(hyp) for hyp in sample['input_ids']]) for sample in batch] # (B, 4, L)

    attention_mask = [torch.tensor([1 for _ in single_id]) for single_id in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value = 0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value = 0)

    refs = [torch.tensor(sample['labels']) for sample in batch]

    top_hyps = [sample['top_hyp'] for sample in batch]

    labels = pad_sequence(refs, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "top_hyp": top_hyps,
        "refs": refs
    }

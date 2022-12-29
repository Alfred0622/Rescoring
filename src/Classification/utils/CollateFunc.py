import torch
from torch.nn.utils.rnn import pad_sequence

def trainBatch(sample):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []
    
    for data in sample:
        input_ids.append(torch.tensor(data['input_ids'], dtype = torch.long))
        token_type_ids.append(torch.tensor(data['token_type_ids'], dtype = torch.long))
        attention_mask.append(torch.tensor(data['attention_mask'], dtype = torch.long))

        labels.append(data['labels'])
    
    input_ids = pad_sequence(input_ids, batch_first = True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    labels = torch.tensor(labels, dtype = torch.long)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
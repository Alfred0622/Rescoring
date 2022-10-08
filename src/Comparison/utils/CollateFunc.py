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

    # for this function, recogBatch will be batch size = 1
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
        
# def bertCompareBatch(sample):
#     # For training set of Comparision

#     tokens = []
#     labels = []
#     segs = []
#     masks = []

#     for s in sample:
#         tokens.append(torch.tensor(s[0]))
#         labels.append(torch.tensor(s[1]))

#         first_sep = s[0].index(102)
#         seg = torch.zeros(len(s[0]))
#         seg[first_sep + 1 :] = 1
#         segs.append(seg)
#         mask = torch.ones(len(s[0]))
#         masks.append(mask)

#     tokens = pad_sequence(tokens, batch_first = True)
#     segs = pad_sequence(segs, batch_first = True, padding_value = 1)
#     masks = pad_sequence(masks, batch_first = True)
#     labels = torch.tensor(labels, dtype = torch.float32)

#     return tokens, segs, masks, labels

# def bertCompareRecogBatch(sample):
#     # For valid, test set of Comparson
#     # sample of dataset include:
#     # [name, token, pair, text, score, err, ref]

#     tokens = []
#     pairs = []
#     segs = []
#     masks = []

#     texts = []
#     first_score = []
#     err = []
#     ref = []

#     for s in sample:
#         name = s[0]

#         for token in s[1]:
#             tokens.append(torch.tensor(token))
#             first_sep = token.index(102)
#             seg = torch.zeros(len(token))
#             seg[first_sep + 1 :] = 1
#             segs.append(seg)
#             mask = torch.ones(len(token))
#             masks.append(mask)

#         pairs += s[2]

#         texts += s[3]
#         first_score += s[4]

#         err += s[5]
#         ref += s[6]

#     tokens = pad_sequence(tokens, batch_first = True)
#     segs = pad_sequence(segs, batch_first = True, padding_value = 1)
#     masks = pad_sequence(masks, batch_first = True)
#     score = torch.zeros(20)

#     return (
#         name,
#         tokens,
#         segs,
#         masks,
#         pairs,
#         texts,
#         first_score,
#         err,
#         ref,
#         score
#     )


  
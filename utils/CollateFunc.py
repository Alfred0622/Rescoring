import torch
import logging
from torch.nn.utils.rnn import pad_sequence

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

    pll = [s[4] for s in sample]
    for p in pll:
        assert len(p) == len(s[0]), f"illegal pll:{p}"
    pll = torch.tensor(pll)

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    # for i, s in enumerate(segs):
    #     segs[i] = torch.tensor(s)

    tokens = pad_sequence(tokens, batch_first=True)

    # segs = pad_sequence(segs, batch_first=True)

    masks = torch.zeros(tokens.shape, dtype=torch.long)
    masks = masks.masked_fill(tokens != 0, 1)

    return tokens, texts, masks, torch.tensor(scores), torch.tensor(cers), pll


 
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


def correctBatch(sample):
    tokens = []
    labels = []
    scores = []
    cers = []
    for s in sample:
        batch = len(s[0])
        for i, t in enumerate(s[0]):
            tokens.append(torch.tensor(t))
            labels.append(torch.tensor(s[1]))

    tokens = pad_sequence(tokens, batch_first = True)
    labels = pad_sequence(labels, batch_first = True, padding_value=-100)

    attention_masks = torch.zeros(tokens.shape)
    attention_masks[tokens != 0] = 1

    return tokens, attention_masks, labels

def correctRecogBatch(sample):
    tokens = []
    errs = []
    texts = []
    score = []
    
    for s in sample :
        tokens += s[0]
        texts += s[3]
    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    
    tokens = torch.tensor(tokens)

    attention_masks = torch.zeros(tokens.shape)
    attention_masks[tokens != 0] = 1

    return tokens, attention_masks, texts


# def createBatch(sample):
#     token_id = [s[0] + s[1] for s in sample]
#     seg_id = [[0 * len(s[0])] + [1 * len(s[1])] for s in sample]

#     for i, token in enumerate(token_id):
#         token_id[i] = torch.tensor(token)
#         seg_id[i] = torch.tensor(seg_id[i])

#     token_id = pad_sequence(token_id, batch_first=True)
#     seg_id = pad_sequence(seg_id, batch_first=True, padding_value=1)

#     attention_mask = torch.zeros(token_id.shape)
#     attention_mask = attention_mask.masked_fill(token_id != 0, 1)

#     labels = [s[2] for s in sample]

#     for i, label in labels:
#         labels[i] = torch.tensor(label)
#     labels = pad_sequence(labels, batch_first=True, padding_value=-100)

#     return token_id, seg_id, attention_mask, labels
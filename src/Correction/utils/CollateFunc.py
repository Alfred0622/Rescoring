import torch
import logging
from torch.nn.utils.rnn import pad_sequence

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

def nBestAlignBatch(sample):
    tokens = [s[0][1:] for s in sample]

    ref_tokens = [s[1] for s in sample]

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)

    for i, t in enumerate(ref_tokens):
        ref_tokens[i] = torch.tensor(t)

    tokens = pad_sequence(tokens, batch_first=True)
    ref_tokens = pad_sequence(ref_tokens, batch_first=True)

    masks = torch.zeros(tokens.shape[:2], dtype=torch.long)

    masks = masks.masked_fill(torch.any(tokens != torch.zeros(tokens.shape[-1])), 1)

    ref = [s[2] for s in sample]

    return tokens, masks, ref_tokens, ref
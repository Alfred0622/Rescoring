import torch
import logging
from torch.nn.utils.rnn import pad_sequence

def bertCompareBatch(sample):
    # For training set of Comparision

    tokens = []
    labels = []
    segs = []
    masks = []

    for s in sample:
        tokens.append(torch.tensor(s[0]))
        labels.append(torch.tensor(s[1]))

        first_sep = s[0].index(102)
        seg = torch.zeros(len(s[0]))
        seg[first_sep + 1 :] = 1
        segs.append(seg)
        mask = torch.ones(len(s[0]))
        masks.append(mask)

    tokens = pad_sequence(tokens, batch_first = True)
    segs = pad_sequence(segs, batch_first = True, padding_value = 1)
    masks = pad_sequence(masks, batch_first = True)
    labels = torch.tensor(labels, dtype = torch.float32)

    return tokens, segs, masks, labels

def bertCompareRecogBatch(sample):
    # For valid, test set of Comparson
    # sample of dataset include:
    # [name, token, pair, text, score, err, ref]

    tokens = []
    pairs = []
    segs = []
    masks = []

    texts = []
    first_score = []
    err = []
    ref = []

    for s in sample:
        name = s[0]

        for token in s[1]:
            tokens.append(torch.tensor(token))
            first_sep = token.index(102)
            seg = torch.zeros(len(token))
            seg[first_sep + 1 :] = 1
            segs.append(seg)
            mask = torch.ones(len(token))
            masks.append(mask)

        pairs += s[2]

        texts += s[3]
        first_score += s[4]

        err += s[5]
        ref += s[6]

    tokens = pad_sequence(tokens, batch_first = True)
    segs = pad_sequence(segs, batch_first = True, padding_value = 1)
    masks = pad_sequence(masks, batch_first = True)
    score = torch.zeros(20)

    return (
        name,
        tokens,
        segs,
        masks,
        pairs,
        texts,
        first_score,
        err,
        ref,
        score
    )


  
import torch
import logging
from torch.nn.utils.rnn import pad_sequence

def correctBatch(sample):
    tokens = []
    labels = []
    scores = []
    cers = []
    for s in sample:
        tokens.append(torch.tensor(s[0]))
        labels.append(torch.tensor(s[1]))
        # for i, t in enumerate(s[0]):
        #     print(f'token:{t}')
        #     tokens.append(torch.tensor(t))
            
    tokens = pad_sequence(tokens, batch_first = True)
    labels = pad_sequence(labels, batch_first = True, padding_value=-100)

    attention_masks = torch.zeros(tokens.shape)
    attention_masks[tokens != 0] = 1

    return tokens, attention_masks, labels

def correctRecogBatch(sample):
    tokens = []
    texts = []
    
    for s in sample :
        tokens.append(torch.tensor(s[0]))
        texts.append(s[1])
    # for i, t in enumerate(tokens):
    #     tokens[i] = torch.tensor(t)
    
    tokens = torch.tensor(tokens)

    attention_masks = torch.zeros(tokens.shape)
    attention_masks[tokens != 0] = 1

    return tokens, attention_masks, texts

def nBestAlignBatch(sample):
    tokens = [s[0] for s in sample]

    ref_tokens = [s[1] for s in sample]

    masks = list()

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
        # logging.warning(f'tokens[i]:{tokens[i]}')
        masks.append(torch.ones(tokens[i].shape[0]))

    for i, t in enumerate(ref_tokens):
        ref_tokens[i] = torch.tensor(t)

    tokens = pad_sequence(tokens, batch_first=True)
    # logging.warning(f'Pad tokens: {tokens}')
    ref_tokens = pad_sequence(ref_tokens, batch_first=True)
    masks = pad_sequence(masks, batch_first = True)

    # masks = torch.zeros(tokens.shape[:2], dtype=torch.long)

    # masks = masks.masked_fill(torch.any(tokens != torch.zeros(tokens.shape[-1])), 1)

    ref = [s[2] for s in sample]

    return tokens, masks, ref_tokens, ref
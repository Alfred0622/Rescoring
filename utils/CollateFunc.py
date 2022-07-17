import torch
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

def bertCompareBatch(sample):
    # For training set of Comparision

    tokens = []
    labels = []
    segs = []
    masks = []

    for s in sample:
        tokens.append(s[0])
        labels.append(s[1])

        first_sep = s[0].index(102)
        seg = torch.zeros(len(s[0]))
        seg[first_sep + 1 :] = 1
        segs.append(seg)
        mask = torch.ones(len(s[0]))
        masks.append(mask)
    tokens = pad_sequence(tokens, batch_first = True)
    segs = pad_sequence(segs, batch_first = True, padding_value = 1)
    masks = pad_sequence(masks, batch_first = True)
    labels = torch.tensor(labels)

    return tokens, segs, masks, labels

def bertCompareRecogBatch(sample):
    # For valid, test set of Comparson
    # sample of dataset include:
    # [token, text, score, err]

    tokens = []
    segs = []
    masks = []
    label = []
    texts = []
    first_score = []
    errs = []

    scores = []

    pairs = [] 
    for s in sample:
        # 1. for every token sequence, concat to every other token
        #    and add a label
        for i, first_seq in s[0]:
            for j, sec_seq in s[0]:
                if (i == j):
                    continue
                concat_seq = first_seq + sec_seq[1:]
                tokens.append(torch.tensor(concat_seq))
                if (i < j): 
                    # if the index of first_seq is smaller than the second
                    label.append(1) 
                    # first_seq is more like oracle than second one 
                else:
                    label.append(0)
                segs.append(
                    torch.tensor(
                        [0 for _ in range(len(first_seq))] + [1 for _ in range(len(sec_seq) - 1)]
                    )
                )
                masks.append(
                    torch.tensor(
                        [1 for _ in range(len(first_seq) + len(sec_seq) - 1)]
                    )
                )
                pairs.append([i, j])
        texts += s[1]
        first_score += s[2]
        errs += s[3]

        scores.append(torch.zeros(len(s[0])))

        # pad sequence
        # pad token, seg, mask to same length
    tokens = pad_sequence(tokens, batch_first = True)
    segs = pad_sequence(segs, batch_first = True, padding_value = 1)
    masks = pad_sequence(masks, batch_first = True)
    label = label.tensor(label)

    cers = torch.tensor(cers)
    errs = torch.tensor(errs)
    pairs = torch.tensor(pairs)

    scores = scores.stack(scores)

    return tokens, segs, masks, first_score, errs, pairs, scores, texts, label


def correctBatch(sample):
    tokens = []
    labels = []
    for s in sample:
        batch = len(s[0])
        max_len = 0
        for i, t in enumerate(s[0]):
            if (len(t)< len(s[1])):
                # pad the input that is shorter than label
                pad_len = len(s[1]) - len(t)
                pad_tokens = t + [0 for _ in range(pad_len)]
            
            else:
                pad_tokens = t
            
            if(len(t) > max_len):
                max_len = len(t)

            tokens.append(torch.tensor(pad_tokens))

        if (len(s[1]) < max_len):
            # pad the label if a sequence that is longer than label exists
            pad_len = max_len - len(s[1])
            label = s[1] + [0 for _ in range(pad_len)]
        else:
            label = s[1]
        for _ in range(batch):
            labels.append(torch.tensor(label))
        
    tokens = pad_sequence(tokens, batch_first = True)
    labels = pad_sequence(labels, batch_first = True)

    return tokens, labels

def correctRecogBatch(sample):
    tokens = []
    errs = []
    texts = []
    score = []
    
    for s in sample :
        tokens += s[0]
        errs += s[2]
        texts += s[3]
        ref = s[4]
        score += s[5]
    for i, t in tokens:
        tokens[i] = torch.tensor(t)
    
    tokens = pad_sequence(tokens, batch_first = True)
    errs = torch.tensor(errs)
    score = torch.tensor(score)


    return tokens, errs, texts, ref, score

        
        
import torch
import logging

def get_sentence_score(logits, input_ids):
    input_ids = torch.roll(input_ids, -1)

    input_ids[:, -1] = 0
    
    cls_index = (input_ids == 101)
    sep_index = (input_ids == 102)
    pad_index = (input_ids == 0)

    temp_index = torch.logical_or(cls_index, sep_index)
    exclude_index = torch.logical_or(temp_index, pad_index)

    score = logits.gather(2, input_ids.unsqueeze(2)).squeeze(-1)
    sum_score = score.sum(dim = -1)
    score[exclude_index] = 0.0
    result = score.sum(dim = -1)

    return result
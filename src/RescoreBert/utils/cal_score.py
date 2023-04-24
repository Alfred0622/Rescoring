import torch
import logging

def get_sentence_score(logits, input_ids, bos = 101, eos = 102, pad = 0):
    input_ids = torch.roll(input_ids, -1) # shift left 1 to
    # Since score at every index represent the probabiility for next tokens

    input_ids[:, -1] = 0

    # print(f'input_ids after roll:{input_ids}')

    # print(f'input_ids:{input_ids}')
    # print(f'scores:{logits.shape}')
    
    cls_index = (input_ids == bos)
    sep_index = (input_ids == eos)
    pad_index = (input_ids == pad)

    temp_index = torch.logical_or(cls_index, sep_index)
    exclude_index = torch.logical_or(temp_index, pad_index)

    score = logits.gather(2, input_ids.unsqueeze(2)).squeeze(-1)
    # print(f'gather score:{score.shape}')
    # sum_score = score.sum(dim = -1)
    # print(f'sum_score:{sum_score}')
    
    score[exclude_index] = 0.0
    result = score.sum(dim = -1)
    # print(f'score exclude special token:{result}')

    # print(f'result:{result.shape}')
    
    return result


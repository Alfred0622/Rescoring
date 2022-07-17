import torch

def get_sentence_score(logits, input_ids):
    non_cls_index = (input_ids != 101)
    non_sep_index = (input_ids != 102)
    non_pad_index = (input_ids != 0)

    temp_logit = torch.logical_and(non_cls_index, non_sep_index)
    token_index = torch.logical_and(temp_logit, non_pad_index)

    score = logit.gather(2, seq.unsqueeze(2)).squeeze(-1)

    sum_score = 0
    last_i = -1
    result = []
    for i, j in true_index:
        if (last_i >= 0 and last_i != i):
            result.append(sum_score)
            sum_score = 0
        print(i)
        sum_score += score[i][j].item()
        last_i = i.item()
    result.append(sum_score)

    return torch.tensor(result)
import torch
from torch.nn.utils.rnn import pad_sequence

def trainBatch(batch):
    input_1 = []
    input_2 = []
    am_score = []
    ctc_score = []
    lm_score = []
    labels = []

    for sample in batch:
        input_1.append(torch.tensor(sample['input_1'], dtype = torch.int))
        input_2.append(torch.tensor(sample['input_2'], dtype = torch.int))

        am_score.append(sample['am_score'])
        ctc_score.append(sample['ctc_score'])
        lm_score.append(sample['lm_score'])
        labels.append(sample['labels'])
    
    input_1 = pad_sequence(input_1, batch_first = True, padding_value = 0)
    input_2 = pad_sequence(input_2, batch_first = True, padding_value = 0)

    am_score = torch.tensor(am_score)
    lm_score = torch.tensor(lm_score)
    ctc_score = torch.tensor(ctc_score)
    labels = torch.tensor(labels)

    return {
        "input_1": input_1,
        "input_2": input_2,
        "am_score": am_score,
        "ctc_score": ctc_score,
        "lm_score": lm_score,
        "labels": labels
    }
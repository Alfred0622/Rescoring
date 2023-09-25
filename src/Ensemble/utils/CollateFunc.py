import torch
from torch.nn.utils.rnn import pad_sequence
def ensembleCollate(batch):
    features = []
    feature_rank = []
    names = []
    nBestIndex = []
    labels = []
    indexes = []

    for sample in batch:
        features += sample['feature']
        feature_rank += sample['feature_rank']
        names += [sample['name'] for _ in range(len(sample['feature']))]
        nBestIndex.append(len(sample['feature']))
        label = [0 for _ in range(int(sample['nbest']))]
        if (int(sample['label']) >= len(label)):
            print(f"label:{int(sample['label'])}")
        label[int(sample['label'])] = 1
        labels += label
        indexes += [i for i in range(len(sample['feature']))]
    
    features = torch.as_tensor(features, dtype = torch.float32)
    feature_rank = torch.as_tensor(feature_rank, dtype = torch.float32)
    nBestIndex = torch.as_tensor(nBestIndex, dtype = torch.int32)
    labels = torch.as_tensor(labels, dtype = torch.int32)

    feature_rank = torch.reciprocal(feature_rank + 1)

    return {
        'name': names,
        'feature': features,
        'feature_rank': feature_rank,
        'nBestIndex': nBestIndex,
        'index': indexes,
        'label': labels
    }

def ensemblePBERTCollate(batch):
    input_ids = []
    attention_mask = []
    features = []
    names = []
    nBestIndex = []
    labels = []
    indexes = []

    for sample in batch:
        input_ids += [torch.as_tensor(s, dtype=torch.int64) for s in sample["input_ids"]]
        attention_mask += [
            torch.as_tensor(s, dtype=torch.int64) for s in sample["attention_mask"]
        ]
        # print(f'sample:{sample["feature"]}')
        features += [torch.as_tensor(s, dtype = torch.float32) for s in sample['feature']]
        names += [sample['name'] for _ in range(len(sample['feature']))]
        nBestIndex.append(len(sample['feature']))
        label = [0 for _ in range(int(sample['nbest']))]
        if (int(sample['label']) >= len(label)):
            print(f"label:{int(sample['label'])}")
        label[int(sample['label'])] = 1
        labels += label
        indexes += [i for i in range(len(sample['feature']))]
    
    input_ids = pad_sequence(input_ids, batch_first = True)
    attention_mask = pad_sequence(attention_mask, batch_first = True)
    features = torch.stack(features)
    nBestIndex = torch.as_tensor(nBestIndex, dtype = torch.int32)
    labels = torch.as_tensor(labels, dtype = torch.int32)

    return {
        "name": names,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        'feature': features,
        # 'feature_rank': feature_rank,
        'nBestIndex': nBestIndex,
        'index': indexes,
        'label': labels
    }
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

class ensembleDataset(Dataset):
    def __init__(self, data_list):
        """
        data_dict: dict
        dict architecture:
        {
            name: {
                    CLM: score,
                    MLM: score,

                  }
        }
        """
        self.data = data_list
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def load_scoreData(data_json, data_dict, nbest, retrieve_num, wer_weight = 1.0, use_Norm = False):

    if (retrieve_num < 0):
        for data in data_json:
            scores = np.asarray(data['rescore'][:nbest]) if 'rescore' in data.keys() else np.asarray(data['rescores'][:nbest])
            if (use_Norm):
                if  (scores.min() == scores.max()):
                    print(f"same:{data['name']} \n Score: {scores}")
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            scores = np.nan_to_num(scores)
            scores = scores * wer_weight
            score_tensor = torch.from_numpy(scores)
            score_rank = torch.argsort(score_tensor, dim = -1,descending=True).tolist()
            for i, (score, rank) in enumerate(zip(scores, score_rank)):
                data_dict[data['name']]['feature'][i].append(score)
                if ('feature_rank'in data_dict[data['name']].keys()):
                    data_dict[data['name']]['feature_rank'][i].append(rank*wer_weight)
    else:
        name_to_score = dict()
        for data in data_json:
            name_to_score[data['name']] = data['rescore'] if 'rescore' in data.keys() else np.asarray(data['rescores'])
        for name in data_dict.keys():
            scores = np.asarray(name_to_score[name][:nbest]) 
            if (use_Norm):
                if  (scores.min() == scores.max()):
                    print(f"same:{data['name']} \n Score: {scores}")
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            scores = np.nan_to_num(scores)
            scores = scores * wer_weight
            score_tensor = torch.from_numpy(scores)
            score_rank = torch.argsort(score_tensor, dim = -1,descending=True).tolist()
            for i, (score, rank) in enumerate(zip(scores, score_rank)):
                data_dict[name]['feature'][i].append(score)
                if ('feature_rank'in data_dict[name].keys()):
                    data_dict[name]['feature_rank'][i].append(rank*wer_weight)

    return data_dict

def prepare_ensemble_dataset(data_dict, topk = 10):
    data_list = []
    for key in tqdm(data_dict.keys()):
        wers = torch.as_tensor(data_dict[key]['wer'][:topk])
        min_index = torch.argmin(wers).item()
        data_list.append(
            {
                'name': key,
                'feature': data_dict[key]['feature'],
                'feature_rank': data_dict[key]['feature_rank'],
                'wers': data_dict[key]['wer'],
                'label': min_index,
                'nbest': len(data_dict[key]['feature'])
            }
        )
    
    return ensembleDataset(data_list)

def prepare_SVM_dataset(data_dict, topk = 10):
    input_list = []
    label_list = []
    for _, key in tqdm(enumerate(data_dict.keys())):
        wers = torch.as_tensor(data_dict[key]['wer'][:topk])
        min_index = torch.argmin(wers).item()
        input_list += data_dict[key]['feature']

        for i, _ in enumerate(data_dict[key]['feature']):
            if (i == min_index):
                # print(f'label')
                label_list.append(1)
            else:
                label_list.append(-1)
    
    return input_list, label_list


def prepare_pBERT_Dataset(score_dict, tokenizer,topk  = 10, sort_by_len = True):
    data_list = list()
    for i , key in tqdm(enumerate(score_dict.keys()), ncols = 100):
        input_ids = []
        attention_mask = []
        features = []

        min_len = 10000
        max_len = -1

        for hyp, feature in zip(score_dict[key]['hyps'][:topk],score_dict[key]['feature']):
            output = tokenizer(hyp)
            input_ids.append(output['input_ids'])
            attention_mask.append(output['attention_mask'])
            features.append(feature)

            if len(output["input_ids"]) > max_len:
                max_len = len(output["input_ids"])
            if len(output["input_ids"]) < min_len:
                min_len = len(output["input_ids"])
        
        nbest = len(score_dict[key]['hyps'][:topk])

        ref = score_dict[key]["ref"]

        wers = torch.as_tensor(score_dict[key]['wer'][:topk])
        min_index = torch.argmin(wers).item()


        data_list.append(
            {
                'name': key,
                "input_ids": input_ids,
                "attention_mask":attention_mask,
                "hyps": score_dict[key]['hyps'][:topk],
                "feature": features,
                "ref": ref,
                "label": min_index,
                "nbest": nbest,
                "max_len": max_len,
                "min_len": min_len
            }
        )
        # if (i > 100):
        #     break
    # print(f'feature:{data_list[-1]["feature"]}')
    if sort_by_len:
        data_list = sorted(data_list, key=lambda x: x["max_len"], reverse=True)
    return ensembleDataset(data_list)



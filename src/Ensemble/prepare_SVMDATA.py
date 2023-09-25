import torch
import json
import sys
import logging
import wandb
from tqdm import tqdm
sys.path.append("../")
from src_utils.LoadConfig import load_config
from model.combineLinear import prepare_CombineLinear
from pathlib import Path
from torch.utils.data import DataLoader
from utils.Datasets import prepare_SVM_dataset, load_scoreData
from utils.CollateFunc import ensembleCollate
from RescoreBert.utils.CollateFunc import  NBestSampler, RescoreBert_BatchSampler
from RescoreBert.utils.PrepareScoring import prepareRescoreDict, prepare_score_dict, calculate_cer, get_result
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


config = "/mnt/disk6/Alfred/Rescoring/src/Ensemble/config/ensembleLinear.yaml"
args, train_args, recog_args = load_config(config)
setting = 'withLM' if args['withLM'] else 'noLM'

# Step 1: Load datasets
# Load every data.json at each task, making 

methods = ['CLM', 'MLM','RescoreBert_MD', 'RescoreBert_MWER',  'RescoreBert_MWED', 'Bert_sem', 'PBERT'] # , 'Bert_alsem' , 'PBert', 'PBert_LSTM', , 'RescoreBert_MWED'
train_score_dict = {}
valid_score_dict = {}
score_dicts = [train_score_dict, valid_score_dict]

dataset = ["train", "dev"]

for task, score_dict in zip(dataset, score_dicts):
    with open(f"/mnt/disk6/Alfred/Rescoring/data/{args['dataset']}/data/{setting}/{task}/data.json") as f:
        print(f'{task}:load org data')
        data_json = json.load(f)
        for data in data_json:
            if (not data['name'] in score_dict.keys()):
                score_dict[data['name']] = dict()
                score_dict[data['name']]['feature'] = [[] for _ in range(int(args['nbest']))]
                score_dict[data['name']]['hyps'] = data['hyps'][:int(args['nbest'])]
                score_dict[data['name']]['ref'] = data['ref']
                score_dict[data['name']]['wer'] = [wer['err'] for wer in data['err'][:int(args['nbest'])]]
            
            for i, score in enumerate(data['am_score'][:args['nbest']]):
                score_dict[data['name']]['feature'][i].append(score)
            for i, score in enumerate(data['ctc_score'][:args['nbest']]):
                score_dict[data['name']]['feature'][i].append(score)
            if (data['lm_score'] is not None):
                for i, score in enumerate(data['lm_score'][:args['nbest']]):
                    score_dict[data['name']]['feature'][i].append(score)
            else:
                for i, score in enumerate(data['am_score'][:args['nbest']]):
                    score_dict[data['name']]['feature'][i].append(0.0)

    print(f'{task}:load rescore data')
    data_path = Path(f"/mnt/disk6/Alfred/Rescoring/data/result/aishell/noLM/{task}/10best")
    for method in methods:
        print(f'{task}: {method} loading')
        ensemble_path = f"{data_path}/{method}/data.json"

        with open(ensemble_path) as f:
            ensemble_data_json = json.load(f)
    
        score_dict = load_scoreData(ensemble_data_json,score_dict)

print(f'# of train:{len(train_score_dict.keys())},\n # of valid:{len(valid_score_dict.keys())}')

(
    index_dict,
    inverse_dict,
    am_scores,
    ctc_scores,
    lm_scores,
    rescores,
    wers,
    hyps,
    refs,
) = prepare_score_dict(data_json, nbest=args["nbest"])
    
# check feature_num:
train_feature_num = -1
for name in train_score_dict.keys():
    if (train_feature_num > 0):
        assert(train_feature_num == len(train_score_dict[name]['feature'][0])), f"{train_feature_num} != {len(train_score_dict[name]['feature'][0])}"
    train_feature_num = len(train_score_dict[name]['feature'][0])
    # take the score part in data_json

valid_feature_num = -1
for name in valid_score_dict.keys():
    if (valid_feature_num > 0):
        assert(valid_feature_num == len(valid_score_dict[name]['feature'][0])), f"{valid_feature_num} != {len(valid_score_dict[name]['feature'][0])}"
    valid_feature_num = len(valid_score_dict[name]['feature'][0])

assert(train_feature_num == valid_feature_num), f"train:{train_feature_num}, valid:{valid_feature_num}"
# concat making a dataset
checkpoint_path = Path(
    f"./checkpoint/{args['dataset']}/Ensemble/SVM/{setting}/{args['nbest']}best/"
)
checkpoint_path.mkdir(parents=True, exist_ok=True)

train_features, train_labels = prepare_SVM_dataset(train_score_dict)
valid_features, valid_labels =  prepare_SVM_dataset(valid_score_dict)

print(f'train_features:{train_features[0]}')
print(f'train_labels:{train_labels[0]}')

from sklearn.datasets import dump_svmlight_file
dump_svmlight_file(X=train_features, y=train_labels, f=f"./data/{args['dataset']}/{setting}/train/SVM_data.dat", zero_based=True) 

# with open(f"./data/{args['dataset']}/{setting}/train/SVM_data.bin", 'w') as f:
#     data_string = ""
#     for label, feature in zip(train_labels, train_features):
#         data_string += f"{label} "
#         for k, num in enumerate(feature):
#             if (num != 0):
#                 data_string += f"{k + 1}:{num}"
        

# features = []
# labels = []

# for name in train_score_dict.keys():
#     features += train_score_dict[name]['feature']
#     temp_label = [0 for _ in range(len(train_score_dict[name]['feature']))]
#     wers = np.array(train_score_dict[name]['wer'])
#     min_index = np.argmin(wers)
#     # print(f'wers:{wers}')
#     # print(f'min_index:{min_index}')
#     temp_label[min_index] = 1

#     labels += temp_label


# model = make_pipeline(StandardScaler(), SVR(C = 1.0 , epsilon=0.2))



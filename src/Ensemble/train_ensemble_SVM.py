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
from libsvm.svmutil import *


config = "/mnt/disk6/Alfred/Rescoring/src/Ensemble/config/ensembleLinear.yaml"
args, train_args, recog_args = load_config(config)
setting = 'withLM' if args['withLM'] else 'noLM'

# Step 1: Load datasets
# Load every data.json at each task, making 

methods = ['CLM', 'MLM','RescoreBert_MD', 'Bert_sem', 'PBERT', 'PBert_LSTM'] # , 'Bert_alsem' , 'PBert','RescoreBert_MWER',  'RescoreBert_MWED' , 'RescoreBert_MWED'
wer_weight = {
    'ASR': 7.34,
    'CLM': 6.05,
    'MLM': 5.17,
    'RescoreBert_MD': 5.29,
    'RescoreBert_MWER': 5.29,
    'RescoreBert_MWED': 5.30,
    'PBERT': 5.08,
    'PBert_LSTM': 5.04,
    'Bert_sem': 5.27
}

weight_list = []
weight_list.append(10 - wer_weight['ASR'])

for name in methods:
    weight_list.append( 10 -  wer_weight[name] )

from torch.nn import Softmax
weights = torch.as_tensor(weight_list)
softmax = Softmax(dim = -1)

weights = softmax(weights)

train_score_dict = {}
valid_score_dict = {}
test_score_dict = {}
score_dicts = [train_score_dict, valid_score_dict, test_score_dict]

train_name = []
valid_name = []
test_name = []
name_dict_list = [train_name, valid_name, test_name]

train_index = []
valid_index = []
test_index = []
index_dict_list = [train_index, valid_index, test_index]

dataset = ["train", "dev", "test"]
select_num = 48000
retrieve_num = select_num

for task, score_dict, name_dict, index_dict in zip(dataset, score_dicts, name_dict_list, index_dict_list):
    with open(f"/mnt/disk6/Alfred/Rescoring/data/{args['dataset']}/data/{setting}/{task}/data.json") as f:
        print(f'{task}:load org data')
        data_json = json.load(f)
        if (task in ['dev', 'test']):
            retrieve_num = -1
        if (retrieve_num < 0):
            for data in data_json:
                if (not data['name'] in score_dict.keys()):
                    score_dict[data['name']] = dict()
                    score_dict[data['name']]['feature'] = [[] for _ in range(int(args['nbest']))]
                    score_dict[data['name']]['hyps'] = data['hyps'][:int(args['nbest'])]
                    score_dict[data['name']]['ref'] = data['ref']
                    score_dict[data['name']]['wer'] = [wer['err'] for wer in data['err'][:int(args['nbest'])]]
                
                for i, score in enumerate(data['am_score'][:args['nbest']]):
                    score_dict[data['name']]['feature'][i].append(score)
                    name_dict.append(data['name'])
                    index_dict.append(i)

                for i, score in enumerate(data['ctc_score'][:args['nbest']]):
                    score_dict[data['name']]['feature'][i].append(score)
                if (data['lm_score'] is not None):
                    for i, score in enumerate(data['lm_score'][:args['nbest']]):
                        score_dict[data['name']]['feature'][i].append(score)
                else:
                    for i, score in enumerate(data['am_score'][:args['nbest']]):
                        score_dict[data['name']]['feature'][i].append(0.0)
        else:
            for data in data_json[:retrieve_num]:
                if (not data['name'] in score_dict.keys()):
                    score_dict[data['name']] = dict()
                    score_dict[data['name']]['feature'] = [[] for _ in range(int(args['nbest']))]
                    score_dict[data['name']]['hyps'] = data['hyps'][:int(args['nbest'])]
                    score_dict[data['name']]['ref'] = data['ref']
                    score_dict[data['name']]['wer'] = [wer['err'] for wer in data['err'][:int(args['nbest'])]]
                
                for i, score in enumerate(data['am_score'][:args['nbest']]):
                    score_dict[data['name']]['feature'][i].append(score)
                    name_dict.append(data['name'])
                    index_dict.append(i)

                for i, score in enumerate(data['ctc_score'][:args['nbest']]):
                    score_dict[data['name']]['feature'][i].append(score)
                if (data['lm_score'] is not None):
                    for i, score in enumerate(data['lm_score'][:args['nbest']]):
                        score_dict[data['name']]['feature'][i].append(score)
                else:
                    for i, score in enumerate(data['am_score'][:args['nbest']]):
                        score_dict[data['name']]['feature'][i].append(0.0)
                
                score_dict[data['name']]['rescore'] = [0.0 for _ in range(len(score_dict[data['name']]['hyps']))]

    print(f'{task}:load rescore data')
    data_path = Path(f"/mnt/disk6/Alfred/Rescoring/data/result/aishell/noLM/{task}/10best")
    for method in methods:
        print(f'{task}: {method} loading')
        ensemble_path = f"{data_path}/{method}/data.json"

        with open(ensemble_path) as f:
            ensemble_data_json = json.load(f)
    
        score_dict = load_scoreData(ensemble_data_json,score_dict,int(args['nbest']) ,retrieve_num = retrieve_num)

print(f'# of train:{len(train_score_dict.keys())},\n # of valid:{len(valid_score_dict.keys())}')
    
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

test_feature_num = -1
for name in test_score_dict.keys():
    if (test_feature_num > 0):
        assert(test_feature_num == len(test_score_dict[name]['feature'][0])), f"{test_feature_num} != {len(test_score_dict[name]['feature'][0])}"
    test_feature_num = len(test_score_dict[name]['feature'][0])

SVM_name = 'SVM'
if (select_num > 0):
    SVM_name += f"_{select_num}"

assert(train_feature_num == valid_feature_num), f"train:{train_feature_num}, valid:{valid_feature_num}"
# concat making a dataset
checkpoint_path = Path(
    f"./checkpoint/{args['dataset']}/Ensemble/{SVM_name}/{setting}/{args['nbest']}best/"
)
checkpoint_path.mkdir(parents=True, exist_ok=True)

train_features, train_labels = prepare_SVM_dataset(train_score_dict)
valid_features, valid_labels =  prepare_SVM_dataset(valid_score_dict)
test_features, test_labels =  prepare_SVM_dataset(test_score_dict)

print(f'train_features:{train_features[0]}')
print(f'train_labels:{train_labels[0]}')

dataset = {
    'feature': train_features,
    'labels': train_labels
}

torch.save(dataset, f"./data/{args['dataset']}/{setting}/train/data.pt")
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
run_train = True
run_predict = True
if (run_train):
    print(f'train_model')

    prob = svm_problem(train_labels, train_features, isKernel=True)
    setting_name = f'-t 1 -c 1 -b 0'
    param = svm_parameter(setting_name)
    save_setting = setting_name.replace('-', '').replace(' ', '_')
    model = svm_train(prob, param)

    print(f"type:{type(model)}")
    if (isinstance(model, float)):
        print(model)

    svm_save_model(f'./checkpoint/aishell/Ensemble/libSVM/train_checkpoint_retrieve{select_num}_{save_setting}.model', model)

if (run_predict):
    tasks = ['dev', 'test']
    score_dicts = [valid_score_dict, test_score_dict]
    predict_featrues = [valid_features, test_features]
    predict_labels = [valid_labels, test_labels]
    predict_names = [valid_name, test_name]
    predict_index = [valid_index, test_index]

    print(f'predict SVM')

    best_am = 0.0
    best_ctc = 0.0 
    best_lm = 0.0 
    best_rescore = 0.0
    if (len(sys.argv) == 2):
        checkpoint_path = sys.argv[1]
        model = svm_load_model(checkpoint_path)
    for task, score_dict, features, labels, name_dict, index_map in zip(tasks, score_dicts, predict_featrues, predict_labels, predict_names, predict_index):
        with open(f"/mnt/disk6/Alfred/Rescoring/data/{args['dataset']}/data/{setting}/{task}/data.json") as f:
            data_json = json.load(f)
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

        label, acc, val = svm_predict(labels, features, model, '-b 0')
        
        for i, score in enumerate(val):
            rescores[index_dict[name_dict[i]]][index_map[i]] = score[0]
        # print(f'rescores:{rescores}')

        if (task in ['dev']):
            print(f'find_weight')
            best_am, best_ctc, best_lm, best_rescore, eval_cer = calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            am_range=[0, 1],
            ctc_range=[0, 1],
            lm_range=[0, 1],
            rescore_range=[0, 1],
            search_step=0.1,
            recog_mode=True,
        )
        

        cer, result_dict = get_result(
            inverse_dict,
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            hyps,
            refs,
            am_weight = best_am,
            ctc_weight = best_ctc,
            lm_weight = best_lm,
            rescore_weight = best_rescore
        )

        print(f'task:{task}, CER = {cer}')

        # print(f'score:{value}')
        # print(f'score:{score}')
    







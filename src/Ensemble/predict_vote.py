import json
import sys
sys.path.append("../")
import numpy as np
from src_utils.LoadConfig import load_config
from utils.Datasets import prepare_ensemble_dataset, load_scoreData
from RescoreBert.utils.PrepareScoring import prepareRescoreDict, prepare_score_dict, calculate_cer, get_result
from pathlib import Path
import torch
from tqdm import tqdm
from jiwer import wer

config = "/mnt/disk6/Alfred/Rescoring/src/Ensemble/config/ensembleLinear.yaml"
args, train_args, recog_args = load_config(config)
setting = 'withLM' if args['withLM'] else 'noLM'

methods = ['CLM', 'MLM','RescoreBert_MD', 'Bert_sem', 'PBERT','PBert_LSTM' ] # 'Bert_alsem','PBert_LSTM', ''RescoreBert_MWED' , 'RescoreBert_MWER', 'Bert_alsem'
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
use_Weight = True
weight_list = []

if (use_Weight):
    weight_list.append(10 - wer_weight['ASR'])

    for name in methods:
        weight_list.append( 10 -  wer_weight[name] )
    
    from torch.nn import Softmax
    weights = torch.as_tensor(weight_list)
    softmax = Softmax(dim = -1)

    weights = softmax(weights)
else:
    weight_list.append(1)
    for name in methods:
        weight_list.append(1)
    weights = torch.as_tensor(weight_list)

valid_score_dict = {}
test_score_dict = {}
score_dicts = [test_score_dict, valid_score_dict]

dataset = ["test", "dev"]

for task, score_dict in zip(dataset, score_dicts):
    with open(f"/mnt/disk6/Alfred/Rescoring/data/{args['dataset']}/data/{setting}/{task}/data.json") as f:
        print(f'{task}:load org data')
        data_json = json.load(f)
        for data in data_json:
            if (not data['name'] in score_dict.keys()):
                score_dict[data['name']] = dict()
                score_dict[data['name']]['feature'] = [[] for _ in range(int(args['nbest']))]
                score_dict[data['name']]['feature_rank'] = [[] for _ in range(int(args['nbest']))]
                score_dict[data['name']]['hyps'] = data['hyps'][:int(args['nbest'])]
                score_dict[data['name']]['ref'] = data['ref']
                score_dict[data['name']]['wer'] = [wer['err'] for wer in data['err']]
            
            am_score = data['am_score'][:args['nbest']]
            am_rank = torch.as_tensor(am_score).argsort(dim = -1, descending=True).tolist()
            for i, (score, rank) in enumerate(zip(am_score, am_rank)):
                score_dict[data['name']]['feature'][i].append(score*weights[0].item())
                score_dict[data['name']]['feature_rank'][i].append(rank*weights[0].item())

            ctc_score = data['ctc_score'][:args['nbest']]
            ctc_rank = torch.as_tensor(ctc_score).argsort(dim = -1, descending=True).tolist()
            for i, (score, rank) in enumerate(zip(ctc_score, ctc_rank)):
                score_dict[data['name']]['feature'][i].append(score*weights[0].item())
                score_dict[data['name']]['feature_rank'][i].append(rank*weights[0].item())

            if (data['lm_score'] is not None):
                lm_score = data['lm_score'][:args['nbest']]
                lm_rank = torch.as_tensor(lm_score).argsort(dim = -1, descending=True).tolist()
                for i, (score, rank) in enumerate(zip(lm_score, lm_rank)):
                    score_dict[data['name']]['feature'][i].append(score*weights[0].item())
                    score_dict[data['name']]['feature_rank'][i].append(rank*weights[0].item())
            else:
                for i, score in enumerate(data['am_score'][:args['nbest']]):
                    score_dict[data['name']]['feature'][i].append(0.0)
                    score_dict[data['name']]['feature_rank'][i].append(1 / (i + 1))


    print(f'{task}:load rescore data')
    data_path = Path(f"/mnt/disk6/Alfred/Rescoring/data/result/aishell/noLM/{task}/10best")
    for method, weight in zip(methods, weights[1:]):
        print(f'{task}: {method} loading')
        ensemble_path = f"{data_path}/{method}/data.json"

        with open(ensemble_path) as f:
            ensemble_data_json = json.load(f)

        score_dict = load_scoreData(
            ensemble_data_json,
            score_dict, 
            nbest=int(args['nbest']), 
            retrieve_num=-1, 
            wer_weight=weight.item()
        )
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

train_feature_num = -1

print(f"# of test:{len(test_score_dict.keys())}")
print(f"# of valid:{len(valid_score_dict.keys())}")
for name in test_score_dict.keys():
    if (train_feature_num > 0):
        assert(train_feature_num == len(test_score_dict[name]['feature_rank'][0])), f"{train_feature_num} != {len(test_score_dict[name]['feature'][0])}"
    train_feature_num = len(test_score_dict[name]['feature_rank'][0])
    for i in range(len(test_score_dict[name]['feature_rank'])):
        test_score_dict[name]['feature_rank'][i]  = np.asarray(test_score_dict[name]['feature_rank'][i])
    # take the score part in data_json

valid_feature_num = -1
for name in valid_score_dict.keys():
    if (valid_feature_num > 0):
        assert(valid_feature_num == len(valid_score_dict[name]['feature_rank'][0])), f"{valid_feature_num} != {len(valid_score_dict[name]['feature'][0])}"
    valid_feature_num = len(valid_score_dict[name]['feature_rank'][0])
    for i in range(len(valid_score_dict[name]['feature_rank'])):
        valid_score_dict[name]['feature_rank'][i]  = np.asarray(valid_score_dict[name]['feature_rank'][i])

assert(train_feature_num == valid_feature_num), f"train:{train_feature_num}, valid:{valid_feature_num}"

print(f'voting')
for task, score_dict in zip(dataset, score_dicts):
    print(f'{task}')
    hyps = []
    refs = []
    for name in tqdm(score_dict.keys()):
        scores = []
        # for hyp_score in score_dict[name]['feature']:

        #     score = np.sum(hyp_score)
        #     scores.append(score)
        for ranks in score_dict[name]['feature_rank']:
            ranks = (ranks + 1)
            ranks = 5 * ranks
            
            ranks = np.reciprocal(ranks)

            score = np.sum(ranks)
            scores.append(score)
        scores = np.asarray(scores)
        min_index = np.argmax(scores)
        # print(f'scores:{scores}')
        # print(f'min_index:{min_index}')

        hyps.append(score_dict[name]['hyps'][min_index])
        refs.append(score_dict[name]['ref'])
    
    print(f'refs:{refs[0]}')
    print(f'hyps:{hyps[0]}')
    
    print(f'{task}:{wer(refs, hyps)}')

        

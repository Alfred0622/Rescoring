import json
import sys
sys.path.append("../")
import torch
import numpy as np
from numba import jit, njit
from torch.utils.data import DataLoader
import os

from src_utils.LoadConfig import load_config
from utils.Datasets import getRescoreDataset, getRecogDataset
from utils.CollateFunc import RescoreBertBatch, RescoreBertRecogBatch
from utils.PrepareModel import prepare_RescoreBert
from utils.PrepareScoring import (
    calculate_cer_simp,
    calculate_cer,
    get_result_simp,
    get_result,
    prepare_score_dict_simp,
    prepare_score_dict
)

from  pathlib import Path

args, train_args, recog_args = load_config(f'./config/RescoreBert.yaml')

from tqdm import tqdm

checkpoint_path = sys.argv[1]
mode = sys.argv[2]

setting = 'withLM' if (args['withLM']) else 'noLM'

print(f"{args['dataset']} : {setting}")

dev = "dev"

# @njit()
# def calculate_cer(am_scores, ctc_scores, lm_scores, rescores):
#     c = 0
#     s = 0
#     d = 0
#     i = 0
#     for am_score, ctc_score, lm_score, rescore in zip(am_scores, ctc_scores, lm_scores, rescores):

#         total_score = (
#             alpha * am_score + (1 - alpha) * ctc_score + \
#                 beta * lm_score + \
#                 gamma * rescore
#             )

#         max_index = torch.argmax(total_score)

#         c += score_dict[name]['err'][max_index]['hit']
#         s += score_dict[name]['err'][max_index]['sub']
#         d += score_dict[name]['err'][max_index]['del']
#         i += score_dict[name]['err'][max_index]['ins']

#     cer = (s + d + i) / (c + s + d)

#     return cer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = prepare_RescoreBert(args['dataset'], device)
checkpoint = torch.load(checkpoint_path)

model.bert.load_state_dict(checkpoint['bert'])
model.linear.load_state_dict(checkpoint['fc'])

model = model.to(device)
if (torch.cuda.device_count() > 1):
    model = torch.nn.DataParallel(model)

if (args['dataset'] in ['aishell', 'tedlium2', 'tedlium2_conformer']):
    recog_set = ['dev', 'test']
elif (args['dataset'] in ['aishell2']):
    recog_set = ['dev_ios', 'test_ios', 'test_android', 'test_mic']
elif (args['dataset'] in ['csj']):
    recog_set = ['dev', 'eval1', 'eval2', 'eval3']
elif (args['dataset'] in ['librispeech']):
    recog_set = ['valid', 'dev_clean', 'dev_other', 'test_clean', 'test_other']

best_am = 0.0
best_ctc = 0.0
best_lm = 0.0
best_rescore = 0.0

for task in recog_set:
    # get score_dict
    recog_path = f"./data/{args['dataset']}/{setting}/50best/MLM/{task}/rescore_data.json"

    with open(recog_path) as f:
        recog_json = json.load(f)

    # if (args['dataset'] in ['aishell']):
    #     print(f'simp')
    #     index_dict, scores, rescores, wers = prepare_score_dict_simp(recog_json, nbest = args['nbest'])
    
    # else:
    print(f'complex')
    index_dict, inverse_dict,am_scores, ctc_scores, lm_scores, rescores, wers, hyps, refs = prepare_score_dict(recog_json, nbest = args['nbest'])
    
    recog_dataset = getRecogDataset(recog_json, args['dataset'], tokenizer)

    recog_loader = DataLoader(
        recog_dataset,
        batch_size = recog_args['recog_batch'],
        collate_fn=RescoreBertRecogBatch,
    )

    for data in tqdm(recog_loader, ncols = 100):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)

        output = model(
            input_ids = input_ids,
            attention_mask = attention_mask
            )['score']

        for n, (name, index, score) in enumerate(zip(data['name'], data['index'], output)):
            rescores[index_dict[name]][index] += score.item()
    
    if task in ['dev', 'dev_ios', 'dev_clean', 'valid']:
        print(f'find weight')

        # if (args['dataset'] in ['aishell']):
        #     best_alpha, best_beta, min_cer = calculate_cer_simp(
        #         scores,
        #         rescores, 
        #         wers,
        #         alpha_range=[0,10],
        #         beta_range=[0,10],
        #         search_step = 0.1
        #     )
        #     print(f'alpha = {best_alpha}, beta = {best_beta}, min_cer =  {min_cer}')
        
        # else: 
        best_am, best_ctc, best_lm, best_rescore, min_cer = calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            am_range = [0, 1],
            ctc_range = [0, 1],
            lm_range = [0, 1],
            rescore_range = [0, 1],
            search_step = 0.1 
        )              

        print(f'am_weight:{best_am}\n ctc_weight:{best_ctc}\n lm_weight:{best_lm}\n rescore_weight:{best_rescore}\n CER:{min_cer}')

    
    # if (args['dataset'] in ['aishell']):
    #     cer = get_result_simp(
    #         scores, 
    #         rescores, 
    #         wers, 
    #         alpha = best_alpha, 
    #         beta = best_beta
    #     )

    # else:
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
    
    print(f"Dataset:{args['dataset']}")
    print(f'setting:{setting}')
    print(f'task:{task}')
    # print(f"length_norm:{recog_args['length_norm']}")
    print(f'CER : {cer}')

    save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}")
    save_path.mkdir(exist_ok=True, parents=True)

    with open(f'{save_path}/RescoreBert_{mode}_result.json', 'w') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=1) 



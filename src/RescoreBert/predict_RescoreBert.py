import json
import sys
import torch
import numpy as np
from numba import jit, njit
from torch.utils.data import DataLoader
import os

from utils.LoadConfig import load_config
from utils.Datasets import getRescoreDataset, getRecogDataset
from utils.CollateFunc import RescoreBertBatch, RescoreBertRecogBatch
from utils.PrepareModel import prepare_RescoreBert

args, train_args, recog_args = load_config(f'./config/aishell2_RescoreBert.yaml')

from tqdm import tqdm

checkpoint_path = sys.argv[1]

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

if (args['dataset'] in ['aishell', 'tedlium2', 'tedlium2_conformer']):
    recog_set = ['dev', 'test']
elif (args['dataset'] in ['aishell2']):
    recog_set = ['dev_ios', 'test_mic', 'test_mic', 'test_android']
elif (args['dataset'] in ['csj']):
    recog_set = ['dev', 'eval1', 'eval2', 'eval3']

best_alpha = 0.0
best_beta = 0.0
best_gamma = 0.0

for task in recog_set:
    # get score_dict
    recog_path = f"./data/{args['dataset']}/{setting}/50best/MLM/{task}/rescore_data.json"

    with open(recog_path) as f:
        recog_json = json.load(f)
    
    score_dict = dict() # name : index
    am_score = []
    lm_score = []
    ctc_score = []

    scores = []
    rescores = []

    wers = []

    simp_flag = False
    for i, name in enumerate(recog_json.keys()):
        if (name not in score_dict.keys()):
            score_dict[name] = i
            if ('score' in recog_json[name].keys()):
                scores.append(
                    np.array(recog_json[name]['score'][:args["nbest"]])
                )

                lm_score.append(
                    np.array(
                        recog_json[name]['lm_score'][:args["nbest"]], dtype = np.float64
                    ) if (len(recog_json[name]['lm_score']) > 0) else np.zeros(scores[-1].shape[0])
                )

                rescores.append(np.zeros(scores[-1].shape))

                simp_flag = True

            else:    
                am_score.append(
                    np.array(
                        recog_json[name]['am_score'][:args["nbest"]], dtype = np.float64
                    ) if ('am_score' in recog_json.keys()) else torch.zeros(min(args['nbest'], len(recog_json[name]['hyps'])))
                )

                ctc_score.append(
                    np.array(
                        recog_json[name]['ctc_score'][:args["nbest"]], dtype = np.float64
                    ) if ('ctc_score' in recog_json.keys()) else np.zeros(am_score[-1].shape[0])
                )

                lm_score.append(
                    np.array(
                        recog_json[name]['lm_score'][:args["nbest"]], dtype = np.float64
                    ) if (len(recog_json[name]['lm_score']) > 0) else np.zeros(am_score[-1].shape[0])
                )

                rescores.append(np.zeros(am_score[-1].shape))
            
            wer = np.stack(
                [
                    np.array(
                        [v for v in err_dict.values()]
                    ) for err_dict in recog_json[name]['err']
                ] , axis = 0
            ) # [[c,s,d,i], [c, s,d,i], .....] for 50best
            
            wers.append(wer) # for evary utterance

            # score_dict[name]['Rescore'] = np.zeros(score_dict[name]['am_score'].shape[0])
            # score_dict[name]['hyp'] = recog_json[name]['hyps']
            # score_dict[name]['ref'] = recog_json[name]['ref']
    
    recog_dataset = getRecogDataset(recog_json, args['dataset'], tokenizer)

    recog_loader = DataLoader(
        recog_dataset,
        batch_size = recog_args['batch'],
        collate_fn=RescoreBertRecogBatch,

    )

    for data in tqdm(recog_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)

        output = model(
            input_ids = input_ids,
            attention_mask = attention_mask
            )['score']

        for n, (name, index, score) in enumerate(zip(data['name'], data['index'], output)):
                rescores[score_dict[name]][index] += score.item()
    
    if task in ['dev', 'dev_ios', 'dev_clean']:
        print(f'find weight')

        if (simp_flag):
            print(len(scores))
            print(len(rescores))
            print(len(wers))
            best_alpha, best_beta, min_cer = calculate_cer_simp(scores, rescores, wers)
            print(f'alpha = {best_alpha}, beta = {best_beta}, min_cer =  {min_cer}')
        
        else: pass
    
    else:

        if (simp_flag):
            print(f'gets result with alpha = {best_alpha}, beta = {best_beta}')
            cer = get_result_simp(scores, rescores, wers, best_alpha, best_beta)

            print(f"{args['dataset']} {setting} {task} CER:{cer}")           
            
        # for name in score_dict.keys():
        #     am_score = score_dict[name]['am_score']
        #     ctc_score = score_dict[name]['ctc_score']
        #     lm_score = score_dict[name]['lm_score']
        #     rescore = score_dict[name]['Rescore']

        #     total_score = (
        #         best_alpha * am_score + (1 - best_alpha) * ctc_score + \
        #         best_beta * lm_score + \
        #         best_gamma * rescore
        #     )

        #     max_index = torch.argmax(total_score)

        #     c += score_dict[name]['err'][max_index]['hit']
        #     s += score_dict[name]['err'][max_index]['sub']
        #     d += score_dict[name]['err'][max_index]['del']
        #     i += score_dict[name]['err'][max_index]['ins']
        # cer = (s + d + i) / (c + s + d)

        # print(f"{args['dataset']} {setting} {task} CER:{cer}")                    
    


            




import json
import torch
import numpy as np
from tqdm import tqdm

# Models
from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx

from utils.Datasets import get_Dataset
from src_utils.LoadConfig import load_config
from utils.FindWeight import find_weight_simp

ctxs = [mx.gpu(0)]

args, train_args, recog_args = load_config(f'./config/mlm.yaml')

setting = 'withLM' if args['withLM'] else 'noLM'

if args['dataset'] in ['aishell', 'aishell2']:
    pretrain_name = 'bert-base-chinese'
elif (args['dataset'] in ['tedlium2', 'tedlium2_conformer', 'librispeech']):
    pretrain_name = 'bert-base-uncased'

model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-chinese')
scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)

if (args['dataset'] in ['aishell', 'tedlium2']):
    recog_set = ['dev', 'test']
elif (args['dataset'] in ['csj']):
    pass
elif (args['dataset'] in ['librispeech']):
    pass
elif (args['dataset'] in ['aishell2']):
    pass

min_CER = 1e8
best_weight = 0

for task in recog_set:
    rescore_list = list()
    with open(f"../../data/{args['dataset']}/data/{setting}/{task}/data.json") as f:
        data_json = json.load(f)
    
    for data in data_json:
        rescore = scorer.score_sentences(data['hyps'])

        data['rescore'] = rescore

        rescore_list.append(data)
    
    with open(f"./data/{args['dataset']}/{setting}/50best/MLM/{task}/Paper_rescore_data.json", 'w') as f:
        json.dump(rescore_list, f, ensure_ascii=False, indent = 2)
    
    if (task in ['dev']):
        for weight in np.arange(0, 1, step = 0.01):
            c = 0
            s = 0
            i = 0
            d = 0
            for data in rescore_list:
                data['rescore'] = torch.tensor(data['rescore'], dtype = torch.float64)
                data['score'] = torch.tensor(data['score'], dtype = torch.float64)

                score = (1 - weight) * data['score'] + weight * data['rescore']
            
                best_index = torch.argmax(score)

                c += data['err'][best_index]['hit']
                s += data['err'][best_index]['sub']
                d += data['err'][best_index]['del']
                i += data['err'][best_index]['ins']
            
            CER = (s + d + i) / (c + s + d)

            if (CER < min_CER):
                best_weight = weight
                min_CER = CER
            
            print(f'weight:{weight}: CER:{CER}')

        print(f'best_weight: {best_weight}, CER:{min_CER}')
    
    c = 0
    s = 0
    i = 0
    d = 0
    for data in rescore_list:
        data['rescore'] = torch.tensor(data['rescore'], dtype = torch.float64)
        data['score'] = torch.tensor(data['score'], dtype = torch.float64)
        
        score = (1 - best_weight) * data['score'] + best_weight * data['rescore']
    
        best_index = torch.argmax(score)

        c += data['err'][best_index]['hit']
        s += data['err'][best_index]['sub']
        d += data['err'][best_index]['del']
        i += data['err'][best_index]['ins']
    
    CER = (s + d + i) / (c + s + d)

    print(f'{task} CER: {CER}')



        

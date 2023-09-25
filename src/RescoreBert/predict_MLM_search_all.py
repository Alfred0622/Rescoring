import sys

sys.path.append("..")
from multiprocessing.spawn import prepare
import torch
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax
from utils.Datasets import get_Dataset
from utils.CollateFunc import recogMLMBatch
from utils.PrepareModel import prepare_GPT2, prepare_MLM
from src_utils.LoadConfig import load_config
from utils.FindWeight import find_weight, find_weight_simp
from utils.Datasets import get_mlm_dataset
from utils.PrepareScoring import (
    prepare_score_dict_simp,
    prepare_score_dict,
    calculate_cer_simp,
    calculate_cer,
    get_result_simp,
    get_result
)
from utils.DataPara import BalancedDataParallel
import gc

from jiwer import wer
import time

# checkpoint_path = sys.argv[1]
# mode = sys.argv[2] # best or last
# print(checkpoint_path)

config = "./config/mlm.yaml"
args, train_args, recog_args = load_config(config)

setting = 'withLM' if args['withLM'] else 'noLM'

print(f'withLM:{setting}')

if (args['dataset'] in ['aishell2']):
    if (args['withLM']):
        ctc_weight = 0.7
    else:
        ctc_weight = 0.5
elif (args['dataset'] in ['aishell']):
    ctc_weight = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = prepare_MLM(args['dataset'], device)

bos = tokenizer.cls_token
eos = tokenizer.sep_token
pad = tokenizer.pad_token

dataset_steps = {
    'aishell': [470 * i for i in range(1, 11)],
    'aishell2': [3942, 7885, 11827, 15770, 19712, 23655, 27597, 31540, 35482, 39420],
    'csj': [1575 * i for i in range(1, 11)],
    'tedlium2': [362,725,1087,1450,1812,2175,2537,2900,3262,3620],
    'librispeech' : [1099 * i for  i in range(1,11)]
}

checkpoint_paths = [
    f"./checkpoint/{args['dataset']}/MLM/checkpoint-{step}/pytorch_model.bin" 
    for step in dataset_steps[args['dataset']]
]
batch_size = recog_args['batch_size']


if (args['dataset'] in ['aishell', 'tedlium2_conformer']):
    recog_set = ['dev', 'test']
elif (args['dataset'] in ['tedlium2']):
    recog_set = ['dev', 'test']
elif (args['dataset'] in ['csj']):
    recog_set = ['dev', 'eval1', 'eval2', 'eval3']
elif (args['dataset'] in ['aishell2']):
    recog_set = ['dev_ios', 'test_ios','test_mic', 'test_android']
elif (args['dataset'] in ['librispeech']):
    recog_set = ['valid', 'dev_clean', 'dev_other','test_clean', 'test_other']

dataset_task = dict()
score_dicts = dict()

for task in recog_set:
    print(task)
    total_time = 0.0
    json_file = f"../../data/{args['dataset']}/data/{setting}/{task}/data.json"

    with open(json_file) as f:
        data_json = json.load(f)
    
    index_dict,inverse_dict, am_scores, ctc_scores, lm_scores, rescores, wers, hyps, refs = prepare_score_dict(
            data_json, nbest = args['nbest']
        )
    

    print(f"{args['dataset']} {task} : {len(data_json)}")
    dataset = get_mlm_dataset(data_json, tokenizer, dataset = args['dataset'], topk = args['nbest'])

    dataset_task[task] = dataset
    score_dicts[task] = {
        'index_dict': index_dict,
        "inverse_dict": inverse_dict, 
        "am_scores" : am_scores, 
        "ctc_scores":ctc_scores,
        "lm_scores":lm_scores, 
        "rescores":rescores, 
        "wers":wers, 
        "hyps":hyps, 
        "refs":refs
    }

for task in recog_set:
    print(f"{task} : {len(dataset_task[task])}")
    print(f"{task} score_dict : {len(score_dicts[task]['index_dict'].keys())}")

flush_dict = score_dicts.copy()
cer_list = []

for checkpoint_path in checkpoint_paths:
    epoch_cer_dict = dict()
    score_dicts = flush_dict.copy()
    print(f'checkpoint:{checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    print(f'loading state_dict')
    model.load_state_dict(checkpoint)
    model = model.to(device)
    if (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)

    model.eval()

    best_alpha = 0.0
    best_beta = 0.0

    best_am_weight = 0.0
    best_ctc_weight = 0.0
    best_lm_weight = 0.0
    best_rescore_weight = 0.0


    for task in recog_set:
        print(task)

        dataloader = DataLoader(
            dataset_task[task],
            batch_size=256,
            collate_fn=recogMLMBatch,
            num_workers=16,
            pin_memory=True,
        )

        index_dict = dict()
        inverse_dict = dict()

        data_len = 0
        name_set = set()

        # for data in data_json:
        #     if (args['nbest'] > len(data['hyps'])):
        #         topk = len(data['hyps'])
        #     else:
        #         topk = args['nbest']
        #     for key in data.keys():
        #         if (key in ['ref', 'name'] or data[key] is None):
        #             continue
        #         data[key] = data[key][:topk]

        #     data_len += topk
        #     data_json[score_dicts[task]["index_dict"][data['name']]]['rescore'] = [0.0 for _ in range(topk)]
        
        print(f"data_len:{data_len}")

        with torch.no_grad():
            for data in tqdm(dataloader, ncols = 100):
                # print(data['input_ids'].shape)
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                score = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    return_dict = True
                ).logits

                score = log_softmax(score, dim = -1)

                for i, (name, seq_index, masked_token ,nbest_index, length) in enumerate(
                    zip(
                        data['name'], data['seq_index'], data['masked_token'], data['nBest_index'], data['length']
                    )
                ):
                    if (seq_index == -1): # empty string
                        # data_json[score_dicts[task]["index_dict"][name]]["rescore"][nbest_index] = np.Ninf
                        # if (task != 'train'  and "train" not in task):
                        score_dicts[task]["rescores"][score_dicts[task]["index_dict"][name]][nbest_index] = np.Ninf
                        
                    else:
                        # data_json[score_dicts[task]["index_dict"][name]]["rescore"][nbest_index] += score[i][seq_index][masked_token].item() / (length if recog_args['length_norm'] else 1)
                        # if (task != 'train' and "train" not in task):
                        score_dicts[task]["rescores"][score_dicts[task]["index_dict"][name]][nbest_index] += score[i][seq_index][masked_token].item() / (length if recog_args['length_norm'] else 1)
                    
                    name_set.add(name)

            
        if (task in ['dev', 'dev_trim', 'dev_ios', 'valid']):
            print(f'Find Weight')

            best_am_weight, best_ctc_weight, best_lm_weight, best_rescore_weight,cer = calculate_cer(
                score_dicts[task]["am_scores"],
                score_dicts[task]["ctc_scores"],
                score_dicts[task]["lm_scores"],
                score_dicts[task]["rescores"],
                score_dicts[task]["wers"],
                am_range = [0, 1],
                ctc_range = [0, 1],
                lm_range = [0, 1],
                rescore_range = [0, 1]
            )

        cer, result_dict = get_result(
            index_dict = score_dicts[task]["inverse_dict"],
            am_scores = score_dicts[task]["am_scores"],
            ctc_scores = score_dicts[task]["ctc_scores"],
            lm_scores = score_dicts[task]["lm_scores"],
            rescores = score_dicts[task]["rescores"],
            wers = score_dicts[task]["wers"],
            hyps = score_dicts[task]["hyps"],
            refs = score_dicts[task]["refs"],
            am_weight = best_am_weight,
            ctc_weight = best_ctc_weight,
            lm_weight = best_lm_weight,
            rescore_weight = best_rescore_weight
        )

        epoch_cer_dict[task] = cer
    cer_list.append(epoch_cer_dict)

for i, cers in enumerate(cer_list):
    print(f'epoch:{i + 1}')

    for key in cers.keys():
        print(f'{key} : {cers[key]}')




import json
import sys
sys.path.append("../")
import torch
import numpy as np
from numba import jit, njit
from torch.utils.data import DataLoader
import os

from src_utils.LoadConfig import load_config
from utils.Datasets import prepareListwiseDataset
from utils.CollateFunc import NBestSampler, BatchSampler, crossNBestBatch
from utils.PrepareModel import prepareNBestCrossBert
from utils.PrepareScoring import (
    calculate_cer,
    get_result,
    prepare_score_dict,
    calculate_cerOnRank, 
    get_resultOnRank
)
from src_utils.get_recog_set import get_recog_set 
from pathlib import Path
from tqdm import tqdm
import random

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

config_path = "./config/NBestCrossBert.yaml"
args, train_args, recog_args = load_config(config_path)
mode = "Normal"
if (train_args['useNBestCross']):
    if (train_args['trainAttendWeight']):
        mode = "CrossAttend_TrainWeight"
    else:
        mode = "CrossAttend"

checkpoint_path = sys.argv[1]
mode = sys.argv[2]

setting = 'withLM' if (args['withLM']) else 'noLM'
print(f"{args['dataset']} : {setting}")

if (checkpoint_path != "no"):
    checkpoint = torch.load(checkpoint_path)
    if ('train_args' in checkpoint.keys()):
        train_args = checkpoint['train_args']

for k in train_args.keys():
    print(f'{k}:{train_args[k]}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = prepareNBestCrossBert(
    dataset = args['dataset'], 
    device = device, 
    lstm_dim = train_args['lstm_embedding'],
    useNbestCross = train_args['useNBestCross'], 
    trainAttendWeight = False,
    concatCLS=train_args['concatCLS'],
    fuseType = train_args['fuseType'],
)

model = model.to(device)
model.eval()

# for key in checkpoint['model'].keys():
#     print(f"ok {key}, {checkpoint['model'][key].value.shape}")
# print(f"checkpoint:{checkpoint['model'].keys()}")

if (checkpoint_path != "no"):
    if ('clsWeight' in checkpoint['model'].keys()):

        model.set_weight(checkpoint['model']['clsWeight'], checkpoint['model']['maskWeight'], is_weight = True)
    model.load_state_dict(checkpoint['model'])

recog_set = get_recog_set(args['dataset'])
dev_set = recog_set[0]

best_am = 0.0
best_ctc = 0.0
best_lm = 0.0
best_rescore = 0.0

for task in recog_set:
    recog_path = f"../../data/{args['dataset']}/data/{setting}/{task}/data.json"
    with open(recog_path) as f:
        recog_json = json.load(f)

        index_dict, inverse_dict,am_scores, ctc_scores, lm_scores, rescores, wers, hyps, refs = prepare_score_dict(recog_json, nbest = args['nbest'])
        
        recog_dataset = prepareListwiseDataset(
            recog_json, 
            args['dataset'],
            tokenizer, 
            sort_by_len=True,
            maskEmbedding=train_args['fuseType'] == 'query',
            concatMask=train_args['concatMaskAfter'] if 'concatMaskAfter' in train_args.keys() else False
        )
        recog_sampler = NBestSampler(recog_dataset)
        recog_batch_sampler = BatchSampler(recog_sampler, recog_args['batch'])
        recog_loader = DataLoader(
            dataset = recog_dataset,
            batch_sampler = recog_batch_sampler,
            collate_fn = crossNBestBatch,
            num_workers=16,
        )
        with torch.no_grad():
            for data in tqdm(recog_loader, ncols = 100):
                for key in  data.keys():
                    if (key not in ['name', 'indexes']):
                        data[key] = data[key].to(device)

                output = model(
                    input_ids = data['input_ids'],
                    attention_mask = data['attention_mask'],
                    batch_attention_matrix= data['crossAttentionMask'],
                    am_score = data['am_score'],
                    ctc_score = data['ctc_score'],
                    labels = None,
                    N_best_index= None
                )['score']

                for n, (name, index,  score) in enumerate(zip(data['name'], data['indexes'], output)):
                    rescores[index_dict[name]][index] += score.item()

        if (task == dev_set):
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
        print(f'CER : {cer}')

        save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}")
        save_path.mkdir(exist_ok=True, parents=True)

        with open(f'{save_path}/NBestCrossBert_{mode}_result.json', 'w') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=1) 

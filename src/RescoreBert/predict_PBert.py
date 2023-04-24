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
from utils.CollateFunc import NBestSampler, BatchSampler, PBertBatch
from utils.PrepareModel import preparePBert
from utils.PrepareScoring import (
    calculate_cer,
    get_result,
    prepare_score_dict
)
from src_utils.get_recog_set import get_recog_set 
from pathlib import Path
from tqdm import tqdm

config_path = "./config/PBert.yaml"
args, train_args, recog_args = load_config(config_path)
mode = ""
if (train_args['hard_label']):
    mode = "_HardLabel"
else:
    mode = ""

checkpoint_path = sys.argv[1]
mode = "PBERT" + mode

setting = 'withLM' if (args['withLM']) else 'noLM'
print(f"{args['dataset']} : {setting}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = preparePBert(dataset = args['dataset'], device = device)
model = model.to(device)
model.eval()
checkpoint = torch.load(checkpoint_path)
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
        
        recog_dataset = prepareListwiseDataset(recog_json, tokenizer, sort_by_len=True)
        recog_sampler = NBestSampler(recog_dataset)
        recog_batch_sampler = BatchSampler(recog_sampler, recog_args['batch'])
        recog_loader = DataLoader(
            dataset = recog_dataset,
            batch_sampler = recog_batch_sampler,
            collate_fn = PBertBatch,
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
                    am_score = data['am_score'],
                    ctc_score = data['ctc_score'],
                    N_best_index = None,
                )['score']

                # print(f"output:{output.shape}\n {output}")
                # print(f"name : {data['name']}")
                # print(f"index : {data['indexes']}")

                for n, (name, index,  score) in enumerate(zip(data['name'], data['indexes'], output)):
                    rescores[index_dict[name]][index] += score.item()
                # print(f"rescores: {rescores[index_dict[name]]}")
        
        # print(f"name : {data['name']}")
        # print(f"index : {data['indexes']}")
        # print(f"score:{output}")
        
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
                rescore_range = [0, 5],
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
        # print(f"length_norm:{recog_args['length_norm']}")
        print(f'CER : {cer}')

        save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}")
        save_path.mkdir(exist_ok=True, parents=True)

        with open(f'{save_path}/NBestCrossBert_{mode}_result.json', 'w') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=1) 


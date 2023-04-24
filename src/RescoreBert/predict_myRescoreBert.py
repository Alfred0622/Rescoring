import sys
import numpy as np

from matplotlib.pyplot import get
sys.path.append("..")
# sys.path.append("../..")
from multiprocessing.spawn import prepare
import torch
import json
import random
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax
from utils.Datasets import prepare_myRecogDataset
from utils.CollateFunc import myCollate
from utils.PrepareModel import prepare_myModel
from src_utils.LoadConfig import load_config
from src_utils.get_recog_set import get_recog_set
from utils.FindWeight import find_weight, find_weight_simp
from RescoreBert.utils.PrepareScoring import (
    prepare_score_dict_simp,
    prepare_score_dict,
    calculate_cer,
    calculate_cer_simp,
    get_result,
    get_result_simp
)

checkpoint_path = sys.argv[1]

config = "./config/myRescoreBert.yaml"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args, train_args, recog_args = load_config(config)
setting = 'withLM' if (args['withLM']) else 'noLM'
model, tokenizer, _, _ = prepare_myModel(args['dataset'], train_args['lstm_embedding'], device = device)

checkpoint = torch.load(checkpoint_path)
model.model.load_state_dict(checkpoint['model'])

model = model.to(device)
if (torch.cuda.device_count() > 1):
    model = torch.nn.DataParallel(model)

recog_set = get_recog_set(args['dataset'])

best_am = 0
best_ctc = 0 
best_lm = 0 
best_rescore = 0 
min_cer = 0

for task in recog_set:
    json_file = f"../../data/{args['dataset']}/data/{setting}/{task}/data.json"

    with open(json_file) as f:
        recog_json = json.load(f)
    
    index_dict, inverse_dict,am_scores, ctc_scores, lm_scores, rescores, wers, hyps, refs = prepare_score_dict(recog_json, nbest = args['nbest'])
    
    dataset = prepare_myRecogDataset(recog_json, tokenizer,nbest = args['nbest'])
    dataloader = DataLoader(
        dataset,
        batch_size = int(recog_args['batch']),
        collate_fn=myCollate,
        num_workers = 4 * torch.cuda.device_count(),
        pin_memeory = True
    )

    for i, data in enumerate(dataloader):
        bert_ids = data['bert_ids'].to(device)

        bert_mask = data['bert_mask'].to(device)

        am_scores = data['am_score'].to(device)
        ctc_scores = data['ctc_score'].to(device)

        output = model.recognize(
            bert_ids,
            bert_mask,
            am_scores,
            ctc_scores,
        )

        for n, (name, index, score) in enumerate(zip(data['name'], data['index'], output)):
            rescores[index_dict[name]][index] += score.item()
        
    
    if (task in ['dev', 'dev_ios', 'valid']):
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
    # print(f"length_norm:{recog_args['length_norm']}")
    print(f'CER : {cer}')

    save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}")
    save_path.mkdir(exist_ok=True, parents=True)

    with open(f'{save_path}/MyRescoreAlsem_result.json', 'w') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=1) 
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
from utils.cal_score import get_sentence_score
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax
from utils.Datasets import get_Dataset
from utils.CollateFunc import recogBatch
from utils.PrepareModel import prepare_GPT2
from utils.LoadConfig import load_config
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

args, train_args, recog_args = load_config(f'./config/clm.yaml')
setting = 'withLM' if args['withLM'] else 'noLM'

print(f'setting:{setting}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model, tokenizer = prepare_GPT2(args['dataset'], device)

# if (tokenizer.pad_token is None):
#     tokenizer.pad_token = tokenizer.eos_token
if (tokenizer.cls_token is None):
    tokenizer.cls_token = tokenizer.bos_token
if (tokenizer.sep_token is None):
    tokenizer.sep_token = tokenizer.eos_token

if (args['dataset'] in ["csj"]):
    recog_set = ['dev', 'eval1', 'eval2', 'eval3']
elif (args['dataset'] in ["aishell2"]):
    recog_set = ['dev_ios', 'test_ios', 'test_android', 'test_mic']
else:
    recog_set = ['dev', 'test']


print('get token id')
bos = tokenizer.cls_token_id if (tokenizer.cls_token is not None) else tokenizer.bos_token
eos = tokenizer.sep_token_id if (tokenizer.sep_token is not None) else tokenizer.eos_token
pad = tokenizer.pad_token_id if (tokenizer.pad_token is not None) else 0

print(f'bos:{bos}, eos:{eos}, pad:{pad}')

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model = model.to(device)

random.seed(42)
torch.manual_seed(42)
model.eval()

for task in recog_set:
    with open(f"../../data/{args['dataset']}/data/{setting}/{task}/data.json" , 'r') as f:
        data_json = json.load(f)
        print(f'# of {task} : {len(data_json)}')

    if (args['dataset'] in ['aishell']):
        index_dict, scores, rescores, wers = prepare_score_dict_simp(data_json, int(args['nbest']))
    else:
        index_dict, am_scores, ctc_scores, lm_scores, rescores, wers = prepare_score_dict(data_json, int(args['nbest']))
     
    dataset = get_Dataset(data_json, tokenizer, dataset = args['dataset'] ,topk = args['nbest'], for_train = False, jp_split = args['jp_split'])

    dataloader = DataLoader(
        dataset,
        batch_size=recog_args['batch'],
        collate_fn=recogBatch,
        num_workers=1
    )

    with torch.no_grad():
        for data in tqdm(dataloader, ncols = 100):
            names = data['name']
            indexs = data['index']
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            output = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
            ).logits

            output_scores = log_softmax(output, dim = -1) #(B, L, V)

            score = get_sentence_score(output_scores, input_ids, bos, eos, pad)

            for i, (name, index) in enumerate(zip(names, indexs)):
                rescores[index_dict[name]][index] = score[i].item()

    if (task in ['dev', 'dev_trim', 'dev_ios', 'dev_clean', 'dev_other']):
        print(f'Find Weight')
        if (args['dataset'] in ['aishell']):
            print(f'simp')
            best_alpha, best_beta, min_cer = calculate_cer_simp(scores, rescores, wers, alpha_range = [0, 1], beta_range = [0, 1])
            print(f'best weight: {best_alpha}, {best_beta}, {min_cer}')
        else:
            print('complex')
            best_am_weight, best_ctc_weight, best_lm_weight, best_rescore_weight, cer = calculate_cer(
                am_scores,
                ctc_scores,
                lm_scores,
                rescores,
                wers,
                am_range = np.array([0, 1]),
                ctc_range = np.array([0, 1]),
                lm_range = np.array([0, 1]),
                rescore_range = np.array([0, 1]),
            )
            print(f'best weight:\n am = {best_am_weight},\n ctc = {best_ctc_weight},\n lm = {best_lm_weight}, \n rescore = {best_rescore_weight},\n {cer}')
    
    if (args['dataset'] in ['aishell']):
        am, ctc, lm, res, cer = get_result_simp(scores, rescores, wers, best_alpha, best_beta)
    else:
        cer = get_result(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            best_am_weight,
            best_ctc_weight,
            best_lm_weight,
            best_rescore_weight
        )
    # print(f'{am}, {ctc}, {lm}, {res}')
    print(f"{args['dataset']} {task} -- {setting} CER : {cer}")

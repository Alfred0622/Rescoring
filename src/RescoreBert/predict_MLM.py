import sys

sys.path.append("..")
from multiprocessing.spawn import prepare
import torch
import json
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax
from utils.Datasets import get_Dataset
from utils.CollateFunc import recogMLMBatch
from utils.PrepareModel import prepare_GPT2, prepare_MLM
from utils.LoadConfig import load_config
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

from jiwer import wer

checkpoint_path = sys.argv[1]
print(checkpoint_path)

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

checkpoint = torch.load(checkpoint_path)
print(f'loading state_dict')
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

batch_size = recog_args['batch_size']

for_train = False

if (for_train):
    recog_set = ['train']
elif (args['dataset'] in ['aishell', 'tedlium2_conformer']):
    recog_set = ['dev', 'test']
elif (args['dataset'] in ['tedlium2']):
    recog_set = ['dev', 'test', 'dev_trim']
elif (args['dataset'] in ['csj']):
    recog_set = ['dev', 'eval1', 'eval2', 'eval3']
elif (args['dataset'] in ['aishell2']):
    recog_set = ['dev_ios', 'test_ios','test_mic', 'test_android']
elif (args['dataset'] in ['librispeech']):
    recog_set = ['dev_clean', 'dev_other','test_clean', 'test_other']

best_alpha = 0.0
best_beta = 0.0

best_am_weight = 0.0
best_ctc_weight = 0.0
best_lm_weight = 0.0
best_rescore_weight = 0.0


for task in recog_set:
    print(task)
    json_file = f"../../data/{args['dataset']}/data/{setting}/{task}/data.json"

    with open(json_file) as f:
        data_json = json.load(f)
    
    print(len(data_json))
    dataset = get_mlm_dataset(data_json, tokenizer, dataset = args['dataset'], topk = args['nbest'])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=recogMLMBatch,
    )

    if (args['dataset'] in ['aishell']):
        index_dict, scores, rescores, wers = prepare_score_dict_simp(data_json, nbest = args['nbest'])
    else:
        index_dict, am_scores, ctc_scores, lm_scores, rescores, wers = prepare_score_dict(
            data_json, nbest = args['nbest']
        )
    

    # set score dictionary
    score_dict = dict()
    for data in data_json:
        if (data['name'] not in score_dict.keys()):
            if (args['nbest'] > len(data['hyps'])):
                topk = len(data['hyps'])
            else:
                topk = args['nbest']

            score_dict[data['name']] = dict()
            score_dict[data['name']]['hyps'] = data['hyps'][:topk]
            score_dict[data['name']]['ref'] = data['ref']
            score_dict[data['name']]['Rescore'] = [0 for _ in range(topk)]
            score_dict[data['name']]['err'] = data['err']
            
            if ('am_score' in data.keys() and (data['am_score'] is not None)):
                score_dict[data['name']]['am_score'] = data['am_score'][:topk]
            if ('ctc_score' in data.keys() and (data['ctc_score'] is not None)):
                score_dict[data['name']]['ctc_score'] = data['ctc_score'][:topk]
            if ('lm_score' in data.keys() and (data['lm_score'] is not None) ):
                score_dict[data['name']]['lm_score'] = data['lm_score'][:topk]
            if ('score' in data.keys() and data['score'] is not None):
                score_dict[data['name']]['score'] = data['score'][:topk]

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
            
            # print(f'input_ids.shape:{input_ids.shape}')

            score = log_softmax(score, dim = -1)

            for i, (name, seq_index, masked_token ,nbest_index, length) in enumerate(
                zip(
                    data['name'], data['seq_index'], data['masked_token'], data['nBest_index'], data['length']
                )
            ):
                if (seq_index == -1):
                    rescores[index_dict[name]][nbest_index] += -10000
                    score_dict[name]["Rescore"][nbest_index] += -10000
                else:
                    rescores[index_dict[name]][nbest_index] += score[i][seq_index][masked_token].item() / (length if recog_args['length_norm'] else 1)
                    score_dict[name]["Rescore"][nbest_index] += score[i][seq_index][masked_token].item() / (length if recog_args['length_norm'] else 1)
        
    save_path = Path(f"./data/{args['dataset']}/{setting}/{args['nbest']}best/MLM/{task}")
    save_path.mkdir(parents = True, exist_ok = True)

    with open(f"{save_path}/rescore_data.json", 'w') as f:
        json.dump(score_dict, f, ensure_ascii = False, indent = 4)

    if (task != 'train'):

        if (task in ['dev', 'dev_trim', 'dev_ios', 'dev_clean', 'dev_other']):
            print(f'Find Weight')

            if (args['dataset'] in ['aishell']):
                print('simp')
                best_alpha, best_beta, cer = calculate_cer_simp(
                    scores,
                    rescores, 
                    wers, 
                    alpha_range = [0, 10], 
                    beta_range = [0, 10]
                )
            else:
                print(f'complex')
                best_am_weight, best_ctc_weight, best_lm_weight, best_rescore_weight,cer = calculate_cer(
                    am_scores,
                    ctc_scores,
                    lm_scores,
                    rescores,
                    wers,
                    am_range = [0, 1],
                    ctc_range = [0, 1],
                    lm_range = [0, 1],
                    rescore_range = [0, 1]
                )
        
        
        if (args['dataset'] in ['aishell']):
            cer = get_result_simp(
                scores, 
                rescores, 
                wers, 
                alpha = best_alpha, 
                beta = best_beta
            )
        else:
            cer = get_result(
                am_scores,
                ctc_scores,
                lm_scores,
                rescores,
                wers,
                am_weight = best_am_weight,
                ctc_weight = best_ctc_weight,
                lm_weight = best_lm_weight,
                rescore_weight = best_rescore_weight
            )

        print(f"Dataset:{args['dataset']}")
        print(f'setting:{setting}')
        print(f"length_norm:{recog_args['length_norm']}")
        print(f'CER : {cer}')
        # print(f'JICER:{wer(refs, hyps)}')





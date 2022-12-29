import sys

from matplotlib.pyplot import get
sys.path.append("..")
sys.path.append("../..")
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

checkpoint_path = sys.argv[1]

args, train_args, recog_args = load_config(f'./config/clm.yaml')
setting = 'withLM' if args['withLM'] else 'noLM'

print(f'setting:{setting}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if (args['dataset'] in ['aishell2']):
    if (args['withLM']):
        ctc_weight = 0.7
    else:
        ctc_weight = 0.5
elif (args['dataset'] in ['aishell']):
    ctc_weight = 0

best_weight = 0


for task in recog_set:
    with open(f"../../data/{args['dataset']}/data/{setting}/{task}/data.json" , 'r') as f:
        data_json = json.load(f)
        print(f'# of {task} : {len(data_json)}')
    
    score_dict = dict()

    scores = []
    am_scores = []
    ctc_scores = []
    lm_scores = []

    wers = []

    hyps = []
    for i, data in enumerate(data_json):
        if (data['name'] not in score_dict.keys()):
            if (args['nbest'] > len(data['hyps'])):
                topk = len(data['hyps'])
            else:
                topk = args['nbest']

            score_dict[data['name']] = dict()
            score_dict[data['name']]['hyps'] = data['hyps'][:topk]
            score_dict[data['name']]['ref'] = data['ref']
            if ('score' in data.keys()):
                score_dict[data['name']]['score'] = torch.tensor(data['score'][:topk], dtype = torch.float64)
            score_dict[data['name']]['Rescore'] = torch.zeros(topk)
            score_dict[data['name']]['err'] = data['err'][:topk]
            
            if ('am_score' in data.keys() and data['am_score'] is not None and len(data['am_score']) > 0):
                score_dict[data['name']]['am_score'] = torch.tensor(data['am_score'][:topk], dtype = torch.float64)
                score_dict[data['name']]['ctc_score'] = torch.tensor(data['ctc_score'][:topk], dtype = torch.float64)
            if ('lm_score' in data.keys() and data['lm_score'] is not None and len(data['lm_score']) > 0):
                score_dict[data['name']]['lm_score'] = torch.tensor(data['lm_score'][:topk], dtype = torch.float64)

    dataset = get_Dataset(data_json, tokenizer, dataset = args['dataset'] ,topk = args['nbest'], for_train = False)

    dataloader = DataLoader(
        dataset,
        batch_size=recog_args['batch'],
        collate_fn=recogBatch,
        num_workers=1
    )
    with torch.no_grad():
        for data in tqdm(dataloader):
            names = data['name']
            indexs = data['index']
            # print(f'name:{names}')
            # print(f'index:{indexs}')
            input_ids = data['input_ids'].to(device)
            # print(f'input_ids:{input_ids}')
            attention_mask = data['attention_mask'].to(device)
            output = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                return_dict = True
            ).logits
   
            scores = log_softmax(output, dim = -1) #(B, L, V)

            score = get_sentence_score(scores, input_ids, bos, eos, pad)
        
            # output = model(
            #     input_ids = input_ids,
            #     attention_mask = attention_mask,
            #     labels = input_ids, 
            #     return_dict = True
            # ).loss

            for i, (name, index) in enumerate(zip(names, indexs)):
                score_dict[name]["Rescore"][index] = score[i].item()

    if (task in ['dev', 'dev_trim', 'dev_ios', 'dev_clean', 'dev_other']):
        print(f'Find Weight')
        if ('score' in score_dict[name].keys()):
            print(f'simp')
            best_weight = find_weight_simp(score_dict, bound = [0, 1])
            print(
                    f'\nbest_weight:{best_weight}'
                 )
        else:
            print(f'complex')
            best_lm_weight, best_weight = find_weight(score_dict, bound = [0, 1], ctc_weight = ctc_weight)
            print(
                    f'\nbest_am:{1 - ctc_weight},\n best_ctc:{ctc_weight},\n best_lm:{best_lm_weight}, \nbest_weight:{best_weight}'
                 )

    c = 0
    s = 0
    d = 0
    i = 0
    result_dict = dict()
    for key in score_dict.keys():
        if ('score' in score_dict[key].keys()):
            score = (1 - best_weight) * score_dict[key]['score'] + best_weight * score_dict[key]['Rescore']

        elif (args['withLM']):
            score = ( (1 - ctc_weight) * score_dict[key]['am_score'] + \
                    ctc_weight * score_dict[key]['ctc_score'] ) + \
                    best_lm_weight * score_dict[key]['lm_score'] + \
                    best_weight * score_dict[key]['Rescore']

        else:
            score = (1 - ctc_weight) * score_dict[key]['am_score'] + \
                    ctc_weight * score_dict[key]['ctc_score']  + \
                    best_weight * score_dict[key]['Rescore']

        
        best_index = torch.argmax(score)

        result_dict[key] = {
            "hyp": score_dict[key]['hyps'][best_index],
            "ref": score_dict[key]['ref']
        }

        c += score_dict[key]['err'][best_index]['hit']
        s += score_dict[key]['err'][best_index]['sub']
        d += score_dict[key]['err'][best_index]['del']
        i += score_dict[key]['err'][best_index]['ins']
    
    cer = (s + d + i) / (c + s + d)

    print(f'CER : {cer}')
    save_path = Path(f"./data/{args['dataset']}/{setting}/{args['nbest']}best/CLM")

    save_path.mkdir(parents = True, exist_ok = True)

    with open(f'{save_path}/recog_data.json', 'w') as f:
        json.dump(result_dict, f, ensure_ascii = False, indent = 4)
    

import sys

from matplotlib.pyplot import get
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
from utils.FindWeight import find_weight, find_weight_complex
from utils.Datasets import get_mlm_dataset

checkpoint_path = sys.argv[1]
print(checkpoint_path)

config = "./config/mlm.yaml"
args, train_args, recog_args = load_config(config)

setting = 'withLM' if args['withLM'] else 'noLM'

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

recog_set = ['dev', 'test']

for task in recog_set:
    print(task)
    json_file = f"../../data/{args['dataset']}/data/{setting}/{task}/data.json"

    with open(json_file) as f:
        data_json = json.load(f)
    print(len(data_json))
    dataset = get_mlm_dataset(data_json, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=recogMLMBatch,
        num_workers=4
    )

    # set score dictionary
    score_dict = dict()
    for data in data_json:
        if (data['name'] not in score_dict.keys()):
            if (args['nbest'] > len(data['score'])):
                topk = len(data['score'])
            else:
                topk = args['nbest']

            score_dict[data['name']] = dict()
            score_dict[data['name']]['hyp'] = data['hyp'][:topk]
            score_dict[data['name']]['ref'] = data['ref']
            score_dict[data['name']]['score'] = torch.tensor(data['score'][:topk], dtype = torch.float64)
            score_dict[data['name']]['Rescore'] = torch.zeros(topk)
            score_dict[data['name']]['err'] = data['err']
            
            if ('am_score' in data.keys()):
                score_dict[data['name']]['am_score'] = torch.tensor(data['am_score'][:topk], dtype = torch.float64)
            if ('ctc_score' in data.keys()):
                score_dict[data['name']]['ctc_score'] = torch.tensor(data['ctc_score'][:topk], dtype = torch.float64)
            if ('lm_score' in data.keys()):
                score_dict[data['name']]['lm_score'] = torch.tensor(data['lm_score'][:topk], dtype = torch.float64)
    
    for data in tqdm(dataloader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)

        score = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        ).logits

        score = log_softmax(score, dim = -1)

        for i, (name, seq_index, masked_token ,nbest_index) in enumerate(
            zip(
                data['name'], data['seq_index'], data['masked_token'], data['nBest_index']
            )
        ):
            score_dict[name]["Rescore"][nbest_index] += score[i][seq_index][masked_token].item()
    
    save_dict = score_dict.copy()
    print(id(save_dict))
    print(id(score_dict))

    save_path = Path(f"./data/{args['dataset']}/{setting}/{args['nbest']}best/MLM/{task}")
    save_path.mkdir(parents = True, exist_ok = True)

    for key in save_dict.keys():
        save_dict[key]['score'] = save_dict[key]['score'].tolist()
        save_dict[key]['Rescore'] = save_dict[key]['Rescore'].tolist()
        if ('am_score' in save_dict[key].keys()):
            save_dict[key]['am_score'] = save_dict[key]['am_score'].tolist()
        if ('ctc_score' in save_dict[key].keys()):
            save_dict[key]['ctc_score'] = save_dict[key]['ctc_score'].tolist()
        if ('lm_score' in save_dict[key].keys()):
            save_dict[key]['lm_score'] = save_dict[key]['lm_score'].tolist()

    with open(f"{save_path}/rescore_data.json", 'w') as f:
        json.dump(save_dict, f, ensure_ascii = False, indent = 4)
    
    if (task == 'dev'):
        best_weight = find_weight(score_dict, bound = [1, 10])
    
    c = 0
    s = 0
    d = 0
    i = 0
    result_dict = dict()
    for key in score_dict.keys():
        if (not isinstance(score_dict[key]['score'], torch.Tensor)):
            score_dict[key]['score'] = torch.tensor(score_dict[key]['score'], dtype = torch.float64)
            score_dict[key]['Rescore'] = torch.tensor(score_dict[key]['Rescore'], dtype = torch.float64)
        score = score_dict[key]['score'] + best_weight * score_dict[key]['Rescore']

        best_index = torch.argmax(score)

        result_dict[key] = {
            "hyp": score_dict[key]['hyp'][best_index],
            "ref": score_dict[key]['ref']
        }

        c += score_dict[key]['err'][best_index][0]
        s += score_dict[key]['err'][best_index][1]
        d += score_dict[key]['err'][best_index][2]
        i += score_dict[key]['err'][best_index][3]
    
    cer = (s + d + i) / (c + s + d)

    print(f'CER : {cer}')





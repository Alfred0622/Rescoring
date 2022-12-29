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

for_train = True

if (for_train):
    recog_set = ['train']
elif (args['dataset'] in ['aishell']):
    recog_set = ['dev', 'test']
elif (args['dataset'] in ['tedlium2']):
    recog_set = ['dev', 'test', 'dev_trim']
elif (args['dataset'] in ['csj']):
    recog_set = ['dev', 'eval1', 'eval2', 'eval3']
elif (args['dataset'] in ['aishell2']):
    recog_set = ['dev_ios', 'test_ios','test_mic', 'test_android']
elif (args['dataset'] in ['librispeech']):
    recog_set = ['dev_clean', 'dev_other','test_clean', 'test_other']

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
        for data in tqdm(dataloader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)

            score = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                return_dict = True
            ).logits
            
            # print(f'input_ids.shape:{input_ids.shape}')

            score = log_softmax(score, dim = -1)

            for i, (name, seq_index, masked_token ,nbest_index) in enumerate(
                zip(
                    data['name'], data['seq_index'], data['masked_token'], data['nBest_index']
                )
            ):
                score_dict[name]["Rescore"][nbest_index] += score[i][seq_index][masked_token].item()
        
    save_path = Path(f"./data/{args['dataset']}/{setting}/{args['nbest']}best/MLM/{task}")
    save_path.mkdir(parents = True, exist_ok = True)

    with open(f"{save_path}/rescore_data.json", 'w') as f:
        json.dump(score_dict, f, ensure_ascii = False, indent = 4)

    if (task != 'train'):
        for key in score_dict.keys():
            
            score_dict[key]['Rescore'] = score_dict[key]['Rescore'][:topk]
            if ('am_score' in score_dict[key].keys() and score_dict[key]['am_score'] is not None and len(score_dict[key]['am_score']) > 0):
                score_dict[key]['am_score'] = score_dict[key]['am_score'][:topk]
            if ('ctc_score' in score_dict[key].keys() and score_dict[key]['ctc_score'] is not None and len(score_dict[key]['ctc_score']) > 0):
                score_dict[key]['ctc_score'] = score_dict[key]['ctc_score'][:topk]
            if ('lm_score' in score_dict[key].keys() and score_dict[key]['lm_score'] and len(score_dict[key]['lm_score']) > 0):
                score_dict[key]['lm_score'] = score_dict[key]['lm_score'][:topk]
            if ('score' in score_dict[key].keys() and score_dict[key]['score'] and len(score_dict[key]['score']) > 0):
                score_dict[key]['score'] = score_dict[key]['score'][:topk]

        if (task in ['dev', 'dev_trim', 'dev_ios', 'dev_clean', 'dev_other']):
            print(f'Find Weight')
            if ('score' in score_dict[key].keys()):
                print(f'simp')
                best_weight = find_weight_simp(score_dict, bound = [0, 1])
            else:
                print(f'complex')
                best_lm_weight, best_weight = find_weight(
                    score_dict, 
                    bound = [0, 1], 
                    ctc_weight = ctc_weight
                )
        
        c = 0
        s = 0
        d = 0
        i = 0
        result_dict = dict()

        hyps = list()
        refs = list()

        for key in score_dict.keys():
            if ('score' in  score_dict[key].keys()):
                if (not isinstance(score_dict[key]['score'], torch.Tensor)):
                    score_dict[key]['score'] = torch.tensor(score_dict[key]['score'], dtype = torch.float64)
                    score_dict[key]['Rescore'] = torch.tensor(score_dict[key]['Rescore'], dtype = torch.float64)
                score = (1 - best_weight) * score_dict[key]['score'] + best_weight * score_dict[key]['Rescore']
                # print(f'score:{score}')
            else:
                if (not isinstance(score_dict[key]['am_score'], torch.Tensor)):
                    score_dict[key]['am_score'] = torch.tensor(score_dict[key]['am_score'], dtype = torch.float64)
                if (not isinstance(score_dict[key]['ctc_score'], torch.Tensor)):
                    score_dict[key]['ctc_score'] = torch.tensor(score_dict[key]['ctc_score'], dtype = torch.float64)
                if (len(score_dict[key]['lm_score'] )> 0):
                    if (not isinstance(score_dict[key]['lm_score'], torch.Tensor)):
                        score_dict[key]['lm_score'] = torch.tensor(score_dict[key]['lm_score'], dtype = torch.float64)
                
                score_dict[key]['Rescore'] = torch.tensor(score_dict[key]['Rescore'], dtype = torch.float64)

                if (len(score_dict[key]['lm_score']) > 0):
                    score = ((1 - ctc_weight) * score_dict[key]['am_score'] + \
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

            hyps.append(score_dict[key]['hyps'][best_index])
            refs.append(score_dict[key]['ref'])

            c += score_dict[key]['err'][best_index]['hit']
            s += score_dict[key]['err'][best_index]['sub']
            d += score_dict[key]['err'][best_index]['del']
            i += score_dict[key]['err'][best_index]['ins']
        
        cer = (s + d + i) / (c + s + d)


        print(f'CER : {cer}')
        print(f'JICER:{wer(refs, hyps)}')





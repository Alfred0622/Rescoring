import os
import sys
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader 
from utils.Datasets import get_recogDataset
from utils.LoadConfig import load_config
from utils.CollateFunc import recogBatch
from utils.PrepareModel import prepare_model
from utils.FindWeight import find_weight
# load_config
config_path = "./config/comparison.yaml"
args, train_args, recog_args = load_config(config_path)
setting = "withLM" if args['withLM'] else "noLM"

# prepare_data
if (args['dataset'] == 'csj'):
    recog_set = ['dev','eval1','eval2', 'eval3']
else:
    recog_set = ['dev', 'test']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = prepare_model(args, train_args, device)
checkpoint = torch.load(f'./checkpoint/withLM/20/checkpoint_train_4.pt')
model.model.load_state_dict(checkpoint['state_dict'])
model.linear.load_state_dict(checkpoint['fc_checkpoint'])


am_weight = 1.0
lm_weight = 0.0
rescore_weight = 0.0
for task in recog_set:
    print(f'task:{task}')
    file_name = f"./data/{args['dataset']}/{task}/{setting}/{args['nbest']}best/data.json"
    with open(file_name ,'r') as f:
        data_json = json.load(f)
        
        score_dict = dict()
        for data in data_json:
            if (data['name'] not in score_dict.keys()):
                score_dict[data['name']] = dict()
                score_dict[data['name']]['am_score'] = torch.tensor(data['am_score'][:args["nbest"]], dtype = torch.float32)
                score_dict[data['name']]['ctc_score'] = torch.tensor(data['ctc_score'][:args["nbest"]], dtype = torch.float32)
                score_dict[data['name']]['lm_score'] = torch.tensor(
                    data['lm_score'][:args["nbest"]], dtype = torch.float32
                ) if ("lm_score" in data.keys()) else torch.zeros(args["nbest"])
                
                score_dict[data['name']]['Rescore'] = torch.zeros(args['nbest'])
                score_dict[data['name']]['hyp'] = data['texts']
                score_dict[data['name']]['ref'] = data['ref']
                score_dict[data['name']]['err'] = torch.tensor(data['err'])


        dataset = get_recogDataset(data_json, tokenizer)

        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 4,
            collate_fn = recogBatch,
            num_workers = 1
        )

        for data in tqdm(dataloader):
            
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)

            output = model.recognize(
                input_ids = input_ids,
                token_type_ids = token_type_ids,
                attention_mask = attention_mask
            ).squeeze(-1)
    
            # print(data['pair'])
            # print(output.shape)         
            # print(len(data['name']))
            for n, name in enumerate(data['name']):
                for i, pair in enumerate(data['pair']):
                    first, second = pair
                    # print(f'{first}:{output[i].item()}, {second}:{1 - output[i].item()}')
                    
                    score_dict[name]['Rescore'][first] += output[i].item()
                    score_dict[name]['Rescore'][second] += (1 - output[i].item())

                # print(score_dict[name]['am_score'])
                # print(score_dict[name]['Rescore'])
        
        if (task == 'dev'): # find Best Weight
            print(f'find_best_weight')
            min_cer = 1e8
            for a in range(0, 11): # alpha *  am + (1 - alpha) * ctc_score:
                for b in range(0, 100): # lm_score
                    for g in range(0, 100):
                        print(f"\ralpha = {a}, beta = {b}, gamma = {g}", end="")
                        alpha = a * 0.1
                        beta = b * 0.1
                        gamma = g * 0.1
                        c = 0
                        s = 0
                        d = 0
                        i = 0
                        for name in score_dict.keys():
                            am_score = score_dict[name]['am_score']
                            ctc_score = score_dict[name]['ctc_score']
                            lm_score = score_dict[name]['lm_score']
                            rescore = score_dict[name]['Rescore']
                            score = (
                                alpha * am_score + (1 - alpha) * ctc_score + \
                                beta * lm_score + \
                                gamma * rescore
                            )

                            max_index = torch.argmax(score)

                            best_err = score_dict[name]['err'][max_index]
                            c += best_err[0]
                            s += best_err[1]
                            d += best_err[2]
                            i += best_err[3]

                        cer = (s + d + i) / (c + s + d)
                        if (cer < min_cer):
                            am_weight = alpha
                            lm_weight = beta
                            rescore_weight = gamma
                            min_cer = cer
        
                    for name in score_dict.keys():
                        am_score = score_dict[name]['am_score']
                        ctc_score = score_dict[name]['ctc_score']
                        lm_score = score_dict[name]['lm_score']
                        rescore = score_dict[name]['Rescore']
                        score = (
                                    alpha * am_score + (1 - alpha) * ctc_score + \
                                    beta * lm_score + \
                                    gamma * rescore
                                )

                        max_index = torch.argmax(score)

                        best_err = score_dict[name]['err'][max_index]
                        c += best_err[0]
                        s += best_err[1]
                        d += best_err[2]
                        i += best_err[3]

                        cer = (s + d + i) / (c + s + d)
                        if (cer < min_cer):
                            am_weight = alpha
                            lm_weight = beta
                            rescore_weight = gamma
                            min_cer = cer

        print(f'{am_weight}, {lm_weight}, {rescore_weight}')
        result_dict = dict()
        for name in score_dict.keys():
            if (name not in result_dict.keys()):
                result_dict[name] = dict()

            am_score = score_dict[name]['am_score']
            ctc_score = score_dict[name]['ctc_score']
            lm_score = score_dict[name]['lm_score']
            rescore = score_dict[name]['Rescore']

            # print(f'am_score:{am_score}\n ctc_score:{ctc_score}\n lm_score:{lm_score}\n rescore:{rescore}')

            score = (
                        am_weight * am_score + (1 - am_weight) * ctc_score + \
                        lm_weight * lm_score + \
                        rescore_weight * rescore
                    )
            
            best_idx = torch.argmax(score)

            result_dict[name]['hyp'] = score_dict[name]['hyp'][best_idx]
            result_dict[name]['ref'] = score_dict[name]['ref']
            result_dict[name]['Rescore'] = rescore.tolist()
        
        # print(f'result_dict:{result_dict}')
        with open(f"./data/{args['dataset']}/{task}/{setting}/{args['nbest']}best/rescore_data.json", 'w') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent = 4)

            


            
                        

            
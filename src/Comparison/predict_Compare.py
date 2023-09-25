import os
import sys
sys.path.append("../")
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader 
from utils.Datasets import get_recogDataset, get_recogDatasetFromRaw
from src_utils.LoadConfig import load_config
from utils.CollateFunc import recogBatch, recogWholeBatch
from utils.PrepareModel import prepare_model
from utils.FindWeight import find_weight

from utils.PrepareScoring import (
    prepare_score_dict_simp,
    prepare_score_dict,
    calculate_cer_simp,
    calculate_cer,
    get_result_simp,
    get_result,
    prepare_hyps_dict,
)
from pathlib import Path
import time

# load_config
config_path = "./config/comparison.yaml"
args, train_args, recog_args = load_config(config_path)
setting = "withLM" if args['withLM'] else "noLM"

# prepare_data
if (args['dataset'] == 'csj'):
    recog_set = ['dev','eval1','eval2', 'eval3']
elif (args['dataset'] == 'aishell2'):
    recog_set = ['dev_ios', 'test_ios', 'test_android', 'test_mic']
elif (args['dataset'] == 'librispeech'):
    recog_set = ['valid', 'dev_clean', 'dev_other', 'test_clean', 'test_other']
else:
    recog_set = ['dev', 'test']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = sys.argv[1]
search_all_epoch = True

if (search_all_epoch):
    path_index = checkpoint_path.rindex("/")
    checkpoint_path = checkpoint_path[:path_index]
    checkpoint_paths = [
        f"{checkpoint_path}/checkpoint_train_{i}.pt" 
        for i in range(1,train_args['epoch'] + 1)
    ]
    checkpoint_paths.append(f"{checkpoint_path}/checkpoint_train_bestLoss.pt" )
else:
    checkpoint_paths = [checkpoint_path]

valid_cers = []
test_cers = []

for checkpoint_path in checkpoint_paths:
    print(f'checkpoint:{checkpoint_path}')
    model, tokenizer = prepare_model(args, train_args, device)
    checkpoint = torch.load(checkpoint_path)

    model.bert.load_state_dict(checkpoint['state_dict'])
    model.linear.load_state_dict(checkpoint['fc_checkpoint'])

    best_am = 0.0
    best_ctc = 0.0
    best_lm = 0.0
    best_rescore = 0.0

    print(f"setting:{setting}")
    print(f"nBest:{args['nbest']}")
    result_cer_dict= dict()
    for task in recog_set:
        print(f'setting:{setting}')
        print(f'task:{task}')

        if (args['dataset'] == 'librispeech' and task == 'valid'):
            print(f'load rescore data')
            file_name = f"./data/{args['dataset']}/{task}/{setting}/{args['nbest']}best/test_data.json"
    
        else:
            file_name = f"./data/{args['dataset']}/{task}/{setting}/{args['nbest']}best/data.json"
        hyps_file_name = f"../../data/{args['dataset']}/data/{setting}/{task}/data.json"

        with open(file_name) as f, open(hyps_file_name) as hyp_f:
            data_json = json.load(f)
            data_num = 0
            total_time = 0.0

            index_dict, inverse_dict, am_scores, ctc_scores, lm_scores, rescores, wers = prepare_score_dict(data_json, nbest = args['nbest'])

            if (args['nbest'] == 50):
                dataset = get_recogDatasetFromRaw(data_json, args['dataset'], tokenizer)
            else:
                dataset = get_recogDataset(data_json, args['dataset'], tokenizer)

            data_json = json.load(hyp_f)
            for data in  data_json:
                hyps = data['hyps'][:int(args['nbest'])]
                data_num += len(hyps)

            hyps_dict = prepare_hyps_dict(data_json, nbest = args['nbest'])
            print(f'data_num:{data_num}')

            dataloader = DataLoader(
                dataset = dataset,
                batch_size = recog_args['batch'],
                collate_fn = recogBatch if (args['nbest'] != 50) else recogWholeBatch,
                num_workers = 4
            )

            for data in tqdm(dataloader, ncols = 100):
                input_ids = data['input_ids'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)

                torch.cuda.synchronize()
                t0 = time.time()
                output = model.recognize(
                    input_ids = input_ids,
                    token_type_ids = token_type_ids,
                    attention_mask = attention_mask
                ).squeeze(-1)
                # print(f'output:{type(output)}')
                torch.cuda.synchronize()
                t1 = time.time()
                total_time += (t1-t0)
        
                for i, (name, pair) in enumerate(zip(data['name'], data['pair'])):
                    first, second = pair
                    if (len(output.shape) == 0):
                        rescores[index_dict[name]][first] += output.item()
                        rescores[index_dict[name]][second] += (1 - output.item())
                    else:
                        rescores[index_dict[name]][first] += output[i].item()
                        rescores[index_dict[name]][second] += (1 - output[i].item())

            if (not search_all_epoch):
                rescore_data = []
                for name in index_dict.keys():
                    rescore_data.append(
                        {
                            "name": name,
                            "hyps": hyps_dict[name]['hyps'],
                            "ref": hyps_dict[name]['ref'],
                            "rescore": rescores[index_dict[name]].tolist()
                        }
                    )
                save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}/{args['nbest']}best/Bert_sem")
                save_path.mkdir(exist_ok = True, parents = True)
                with open(f"{save_path}/data.json", 'w') as f:
                    json.dump(rescore_data, f, ensure_ascii=False, indent = 1)

            if (task in ['dev', 'dev_ios', 'valid']): # find Best Weight
                print(f'find_best_weight')

                best_am, best_ctc, best_lm, best_rescore, min_cer = calculate_cer(
                    am_scores,
                    ctc_scores,
                    lm_scores,
                    rescores,
                    wers,
                    am_range = [0, 1],
                    ctc_range = [0, 1],
                    lm_range = [0, 1],
                    rescore_range= [0, 1],
                    search_step = 0.1
                )
                print(f'\nbest weight:\n am = {best_am},\n ctc = {best_ctc},\n lm = {best_lm},\n rescore = {best_rescore},\n CER {min_cer}')

            cer, result_dict = get_result(
                am_scores = am_scores,
                ctc_scores = ctc_scores,
                lm_scores = lm_scores,
                rescores = rescores,
                wers = wers,
                name_dict = inverse_dict,
                hyp_dict = hyps_dict,
                am_weight = best_am,
                ctc_weight = best_ctc,
                lm_weight = best_lm,
                rescore_weight = best_rescore 
            )

            if (not search_all_epoch):
                save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}/{args['nbest']}best/Compare")
                save_path.mkdir(exist_ok = True, parents = True)

                with open(f"{save_path}/analysis.json", 'w') as f:
                    json.dump(result_dict, f, ensure_ascii = False, indent = 1) 
            
                print(f"Dataset:{args['dataset']} {setting} {task} -- CER = {cer}")
                print(f'avg decode time:{total_time / data_num}')
                # print(f'result_dict:{result_dict}')
        result_cer_dict[task] = cer
    
    if (search_all_epoch):
        test_cers.append(result_cer_dict)

if (search_all_epoch):
    for i, (epoch_cer, checkpoint) in enumerate(zip(test_cers, checkpoint_paths)):
        if (i == len(checkpoint_paths) - 1):
            print(f'best')
        else:
            print(f'epoch:{i + 1}')
        print(f'checkpoint:{checkpoint}')
        for data in epoch_cer.keys():
            print(f'{data} : {epoch_cer[data]}')
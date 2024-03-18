import sys
import os
sys.path.append("..")
sys.path.append("../..")
import json
import torch
import logging
from transformers import (
    AutoModelForSeq2SeqLM,
    BertTokenizer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.data import DataLoader

from utils.Datasets import get_dataset
from utils.CollateFunc import recogBatch
from src_utils.LoadConfig import load_config
from utils.PrepareModel import prepare_model
from jiwer import visualize_alignment, process_characters
import time

from pathlib import Path

from jiwer import wer, cer
from tqdm import tqdm

if (len(sys.argv) != 4):
    raise AssertionError("Usage: python predict_nbestAlignConcat.py <mode> <checkpoint_path> <save_name>")

task_name = sys.argv[1]
checkpoint_path = sys.argv[2]
save_name = sys.argv[3]

use_train = False

def predict(model, dataset, tokenizer, loader):
    result = {}
    model.eval()

    print('predict')
    total_time = 0.0

    loop = tqdm(loader, total = len(loader), ncols=100)
    for k, batch in enumerate(loop):
        name = batch["name"]
        input_ids = batch["input_ids"]
        # print(f'input_ids:{input_ids}')
        attention_mask = batch["attention_mask"]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            # torch.cuda.synchronize()
            # t0 = time.time()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            # output = model.predict()
            output = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_length = 150,
                num_beams = 5,
                # early_stopping = True
            )
            end.record()
            torch.cuda.synchronize()
            # t1 = time.time()
            total_time += start.elapsed_time(end)

            output = tokenizer.batch_decode(output, skip_special_tokens = True)
            ref_list = batch['ref_text']
            top_hyps = batch['top_hyp']

            for i, hyp in enumerate(output):
                if (dataset in ['csj', 'aishell', 'aishell2']):
                    hyp = [t for t in "".join(hyp.split())]
                    output[i] = " ".join(hyp)


            for single_name, pred, ref, top_hyp in zip(name, output, ref_list, top_hyps):
                if (single_name not in result.keys()):
                    result[single_name] = dict()
                # ref = " ".join([t for t in "".join(ref.split())])
                # top_hyp = " ".join([t for t in "".join(top_hyp.split())])
                result[single_name]["hyp"] = pred.strip()
                result[single_name]["ref"] = ref.strip()
                result[single_name]["top_hyp"] = top_hyp.strip()

                # print(f'output:{pred}')
                # print(f'ref_list:{ref}')
                # print(f'top_hyps:{top_hyp}\n')

                # print(f'hyp:{result[single_name]["hyp"].replace(" ", "_")}\nref:{result[single_name]["ref"].replace(" ", "_")} \ntop_hyp:{result[single_name]["top_hyp"].replace(" ", "_")}\n')
        
        # if (k > 52):
        #     exit()
    return result, total_time

# Predict
if __name__ == '__main__':
    if (task_name == 'plain'):
        config_name = './config/nBestPlain.yaml'
    else:
        config_name = './config/Bart.yaml'
        topk = 1
    
    print(f'config:{config_name}')

    args, train_args, recog_args = load_config(config_name)
    setting =  'withLM' if args['withLM'] else 'noLM'
    if (args['dataset'] == 'old_aishell'):
        setting = ""
    
    if (task_name == 'plain'):
        topk = args['nbest']
        print(f"sep_token:{train_args['sep_token']}")

    print(f'setting:{setting}')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if (use_train):
        scoring_set = ['train']
    elif (args['dataset'] in ["csj"]):
        scoring_set = ['dev', 'eval1', 'eval2', 'eval3']
    elif (args['dataset'] in ["aishell2"]):
        scoring_set = ["dev", 'test_ios', 'test_mic', 'test_android']
    elif (args['dataset'] in ['librispeech']):
        scoring_set = ['dev_clean', 'dev_other', 'test_clean', 'test_other']
    else:
        scoring_set = ['dev', 'test']

    model, tokenizer = prepare_model(args['dataset'])
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)

    print(f'param num: {sum(p.numel() for p in model.parameters())}')

    for data_name in scoring_set:
        json_path = f"../../data/{args['dataset']}/data/{setting}/{data_name}/data.json"
        if (args['dataset'] == 'csj'):
            json_path = f"./data/{args['dataset']}/{setting}/{data_name}/data.json"
        print(f"recognizing:{data_name}")
        with open(json_path) as f:
            data_json = json.load(f)
        
        print(f'data_len:{len(data_json)}')
        # exit(0)

        dataset = get_dataset(data_json,args['dataset'] , tokenizer, data_type = train_args['data_type'], sep_token=train_args['sep_token'] ,topk = topk, for_train = False)

        dataloader = DataLoader(
            dataset,
            collate_fn = recogBatch,
            batch_size = recog_args['batch'],
            num_workers = 16
        )

        output, total_time = predict(model, args['dataset'],tokenizer, dataloader)

        print(f'average decode time:{total_time / len(data_json)}')

        if (not os.path.exists(f"./data/{args['dataset']}/{setting}/{data_name}/{args['nbest']}{task_name}/")):
            os.makedirs(f"./data/{args['dataset']}/{setting}/{data_name}/{args['nbest']}{task_name}/")
        with open(
            f"./data/{args['dataset']}/{setting}/{data_name}/{args['nbest']}{task_name}/correct_data.json",
            'w', 
            encoding = 'utf-8'
        ) as f:
            json.dump(output, f, ensure_ascii = False, indent = 4)
        
        ref = []
        hyp = []
        top_1_hyp = []

        result_dict = []

        for data in data_json:

            # print(f"\n Ref:{output[data['name']]['ref']} \n Hyp:{output[data['name']]['hyp']}")
            # exit(0)
            if (args['dataset'] in ['csj']):
                output[data['name']]['hyp'] = " ".join(output[data['name']]['hyp'].replace(" ", ""))
                output[data['name']]['top_hyp'] = " ".join(output[data['name']]['top_hyp'].replace(" ", ""))
                output[data['name']]['ref'] = " ".join(output[data['name']]['ref'].replace(" ", ""))

            ref.append(output[data['name']]['ref'])
            hyp.append(output[data['name']]['hyp'])
            top_1_hyp.append(output[data['name']]['top_hyp'])

            corrupt_flag = "Missed"

            if (output[data['name']]['top_hyp'] == output[data['name']]['ref']):
                if (output[data['name']]['hyp'] != output[data['name']]['ref']):
                    corrupt_flag = "Totally_Corrupt"
                else:
                    corrupt_flag = "Remain_Correct"

            else:
                if (output[data['name']]['hyp'] == output[data['name']]['ref']):
                    corrupt_flag = "Totally_Improve"
                else:
                    top_wer = wer(output[data['name']]['ref'], data['hyps'][0])
                    rerank_wer = wer(output[data['name']]['ref'], output[data['name']]['hyp'])
                    if (top_wer < rerank_wer):
                        corrupt_flag = "Partial_Corrupt"
                    elif (top_wer == rerank_wer):
                        corrupt_flag = "Remain_Incorrect"
                    else:
                        corrupt_flag = "Partial_Improve"

            if (args['dataset'] in ['aishell', 'aishell2','csj']):
                out = process_characters(
                    ["".join(output[data['name']]['ref'].split(' '))],
                    ["".join(output[data['name']]['hyp'].split(' '))]
                )
            else:
                out = process_characters(
                    [output[data['name']]['ref']],
                    [output[data['name']]['hyp']]
                )

            align_result = visualize_alignment(out, show_measures=False, skip_correct=False).split('\n')
            align_ref = align_result[1][5:]
            align_hyp = align_result[2][5:]
            result_tag = align_result[3]
    
            result_dict.append(
                {
                    "org": output[data['name']]['top_hyp'],
                    "hyp": output[data['name']]['hyp'],
                    "ref": output[data['name']]['ref'],
                    "REF": align_ref,
                    "HYP": align_hyp,
                    "TAG": result_tag,
                    "check1": "Correct" if output[data['name']]['hyp'] == output[data['name']]['ref'] else "Wrong",
                    "check2": corrupt_flag
                }
            )

        save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{data_name}/")
        save_path.mkdir(parents = True, exist_ok = True)

        with open(f"{save_path}/{args['nbest']}_{task_name}_{save_name}_Correct_result.json", 'w') as f:
            json.dump(result_dict, f, ensure_ascii = False, indent = 1)
        
        print(f'HYP:{result_dict[-1]["hyp"]}')
        print(f'ORG:{result_dict[-1]["org"]}')
        print(f'REF:{result_dict[-1]["ref"]}')
        print(f'org_wer:{wer(ref, top_1_hyp)}')
        print(f'Correction wer:{wer(ref, hyp)}')
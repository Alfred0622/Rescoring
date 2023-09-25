import json
from jiwer import cer, wer
import random
import argparse
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--dataset', 
    help = "which dataset",
    type = str,
    default= 'aishell'
)

parser.add_argument(
    '-w', '--withLM', 
    help = "withLM or not",
    type = str,
    default= 'noLM'
)

parser.add_argument(
    '-n', '--name', 
    help = "save_name",
    type = str,
    default= 'noName'
)

parser.add_argument(
    '-s', '--seed', 
    help = "seed number",
    type = int,
    default= 42
)
args = parser.parse_args()

# random.seed(args.seed)
# np.random.seed(args.seed)
# print(f"seed : {args.seed}")

seeds = [0,1,42]
result_list = []

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    print(f'seed:{seed}')
    result_dict = {}
    recog_set = ['dev', 'test']
    if (args.dataset in ['aishell2']):
        recog_set = ['dev_ios', 'test_ios', 'test_android', 'test_mic']
    elif (args.dataset in ['csj']):
        recog_set = ['eval1', 'eval2', 'eval3']
    elif (args.dataset in ['librispeech']):
        recog_set = ['dev_clean', 'dev_other', 'test_clean', 'test_other']

    # print(f"../data/{args.dataset}/data/{args.withLM}/dev/data.json")
    for recog_task in recog_set:
        with open(f"../data/{args.dataset}/data/{args.withLM}/{recog_task}/data.json") as f:
            data_list = json.load(f)
            top_hyps = []
            hyps = []
            refs = []
            for data in data_list:
                pickIndex = np.random.randint(low = 0, high = len(data['hyps'][:10]))
                top_hyps.append(data['hyps'][0])
                hyps.append(data['hyps'][pickIndex])
                refs.append(data['ref'])

            # print(f'hyps:{hyps[-4:]}')
            # print(f'refs:{refs[-4:]}')

            print(f"{args.dataset} {args.withLM} {recog_task} : Origin WER = {wer(refs, top_hyps)}")
            print(f"{args.dataset} {args.withLM} {recog_task} : WER = {wer(refs, hyps)}\n\n")
    
        result_dict[recog_task] = wer(refs, hyps)
    
    result_list.append(result_dict)

for seed, result in zip(seeds , result_list):
    print(f'seed:{seed}')
    for key in result.keys():
        print(f'{key}:{result[key]}')
    
    print(f'==================================')


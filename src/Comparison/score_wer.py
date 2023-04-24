import pathlib

from yaml import load
from jiwer import wer, cer
import json
from pathlib import Path
from src_utils.LoadConfig import load_config

config = "./config/comparison.yaml"
args, train_args, recog_args = load_config(config)

setting = 'withLM' if args['withLM'] else "noLM"
print(setting)

if (args['dataset'] in ['csj']):
    recog_set = ['dev', 'eval1', 'eval2', 'eval3']
else:
    recog_set = ['dev', 'test']

for task in recog_set:
    print(f'{task}:{setting}')
    correct_data = Path(f"./data/{args['dataset']}/{task}/{setting}/{args['nbest']}best/rescore_data.json")

    hyps = []
    refs = []
    names = []
    with open(correct_data) as f:
        data_json = json.load(f)
        for name in data_json.keys():
            names.append(name)
            hyps.append(data_json[name]['hyp'])
            refs.append(data_json[name]['ref'])

    
    file_name = ['hyp', 'ref']
    write_list = [hyps, refs]
    for file, w_list in zip(file_name, write_list):
        with open(f"./data/{args['dataset']}/{task}/{setting}/{args['nbest']}best/{file}.trn", 'w') as f:
            for name, word  in zip(names, w_list):
                f.write(f"{word} ({name.replace('-', '_')})\n")
    
    print(f'WER:{wer(refs, hyps)}')
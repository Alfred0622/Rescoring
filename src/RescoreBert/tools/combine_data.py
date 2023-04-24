import json
import torch
import sys
from pathlib import Path

dataset_name = sys.argv[1]

if (dataset_name in ['aishell', 'tedlium2']):
    recog_set = ['dev', 'test', 'train']
elif (dataset_name in ['aishell2']):
    recog_set = ['train', 'dev_ios', 'test_mic', 'test_ios', 'test_android']
elif (dataset_name in ['csj']):
    recog_set = ['train', 'dev', 'eval1', 'eval2', 'eval3']
elif (dataset_name in ['librispeech']):
    recog_set = ['train', 'dev_clean', 'dev_other', 'test_clean', 'test_other']
settings = ['withLM', 'noLM']

for setting in settings:
    for task in recog_set:
        print(f"{setting}:{task}")
        with open(
            f"/mnt/disk6/Alfred/Rescoring/data/{dataset_name}/data/{setting}/{task}/data.json"
        ) as f, open (
            f"../data/{dataset_name}/{setting}/50best/MLM/{task}/rescore_data.json"
        ) as rf:
            data_json = json.load(f)
            rescore_data_json = json.load(rf)

            for data in data_json:
                if (not ('am_score' in rescore_data_json[data['name']].keys())):
                    rescore_data_json[data['name']]['am_score'] = data['am_score']
                if (not ('ctc_score' in rescore_data_json[data['name']].keys())):
                    rescore_data_json[data['name']]['ctc_score'] = data['ctc_score']
                if (not ('lm_score' in rescore_data_json[data['name']].keys())):
                    rescore_data_json[data['name']]['lm_score'] = data['lm_score']

                rescore_data_json[data['name']]['score'] = data['score']
                if ("Rescore" in rescore_data_json[data['name']].keys()):
                    rescore_data_json[data['name']]['rescore'] = rescore_data_json[data['name']]['Rescore']
                    rescore_data_json[data['name']].pop('Rescore')
                
            with open(f"../data/{dataset_name}/{setting}/50best/MLM/{task}/rescore_data.json", 'w') as d:
                json.dump(rescore_data_json, d, ensure_ascii=False, indent = 1)
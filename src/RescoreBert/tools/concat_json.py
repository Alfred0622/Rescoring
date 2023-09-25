import json
import torch
import sys
from pathlib import Path

dataset_name = sys.argv[1]

# if (dataset_name in ['aishell', 'tedlium2']):
#     recog_set = ['dev', 'test', 'train']
# elif (dataset_name in ['aishell2']):
#     recog_set = ['train', 'dev_ios', 'test_mic', 'test_ios', 'test_android']
# elif (dataset_name in ['csj']):
#     recog_set = ['train', 'dev', 'eval1', 'eval2', 'eval3']
# elif (dataset_name in ['librispeech']):
#     recog_set = ['train', 'dev_clean', 'dev_other', 'test_clean', 'test_other']

settings = [ 'noLM']


for setting in settings:
    print(f"{dataset_name} -- {setting}")
    concat_Path = Path(f"/mnt/disk6/Alfred/Rescoring/src/RescoreBert/data/{dataset_name}/{setting}/10best/MLM/train/split32")
    concat_Path.mkdir(parents= True, exist_ok=True)
    with open(
        f"/mnt/disk6/Alfred/Rescoring/src/RescoreBert/data/{dataset_name}/{setting}/10best/MLM/train/rescore_data.json", 'w') as f:
        data_json = []
        for split_num in range(1, 33):
            print(f'train_{split_num}')
            with open(
                f"{concat_Path}/train_{split_num}/rescore_data.json"
            ) as g:
                split_json = json.load(g)
                data_json += split_json
            
        json.dump(data_json, f, ensure_ascii = True, indent = 1)

if (dataset_name == 'librispeech'):
    print(f'combine dev_clean and dev_other')
    for setting in settings:
        with open(f"/mnt/disk6/Alfred/Rescoring/src/RescoreBert/data/{dataset_name}/{setting}/10best/MLM/dev_clean/rescore_data.json") as dev_clean, \
             open(f"/mnt/disk6/Alfred/Rescoring/src/RescoreBert/data/{dataset_name}/{setting}/10best/MLM/dev_other/rescore_data.json") as dev_other:
            clean_json = json.load(dev_clean)
            other_json = json.load(dev_other)

            valid_json = clean_json + other_json

            with open(f"/mnt/disk6/Alfred/Rescoring/src/RescoreBert/data/{dataset_name}/{setting}/10best/MLM/valid/rescore_data.json", 'w') as valid:
                json.dump(valid_json, valid, ensure_ascii=True, indent = 1)

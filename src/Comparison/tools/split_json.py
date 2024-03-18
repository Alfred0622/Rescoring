import json
import os
from pathlib import Path


split_num = 32

dataset = ['aishell2'] # 'aishell2', 'csj',

setting = ['noLM', 'withLM']
for d in dataset:
    for s in setting:
        print(f"{d} : {s}")
        with open(f'../data/{d}/train_recog/{s}/10best/data.json', 'r') as f:
            print(f'json_load')
            json_data = json.load(f)
            total_keys = len(json_data)
            print(f'total_keys:{total_keys}')
            sep_key = total_keys // split_num

            temp_dict = list()

            counter = 1
            for i, data in enumerate(json_data):
                temp_dict.append(data)

                if (counter < split_num):
                    if (i >= sep_key * counter):
                        new_path = Path(f'../data/{d}/train_recog/{s}/10best/split_{split_num}/train_{counter}')
                        new_path.mkdir(parents = True, exist_ok = True)
                        with open(f"{new_path}/data.json", 'w') as g:
                            json.dump(temp_dict, g, ensure_ascii = False, indent = 2)
                            print(f"len_{counter}:{len(temp_dict)}")
                            temp_dict = list()
                        counter += 1
                
                elif (counter == split_num):
                    if (i == total_keys - 1):
                        new_path = Path(f'../data/{d}/train_recog/{s}/10best/split_{split_num}/train_{counter}')
                        new_path.mkdir(parents = True, exist_ok = True)
                        with open(f"{new_path}/data.json", 'w') as g:
                            json.dump(temp_dict, g, ensure_ascii = False, indent = 2)
                            print(f"len_{counter}:{len(temp_dict)}")
                            temp_dict = list()
                del json_data[i]

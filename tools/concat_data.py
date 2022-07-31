# concat the token sequence for comparison
import json
from tqdm import tqdm
import os
import random

random.seed(42)
setting = ['noLM', 'withLM']

for s in setting:
    print(f"file: /mnt/disk3/Alfred/Rescoring/data/aishell/train/token/token_{s}_50best.json")
    with open(f'/mnt/disk3/Alfred/Rescoring/data/aishell/train/token/token_{s}_50best.json') as f:
        token_file = json.load(f)
        total_data = len(token_file)
        # file_num = 64
        # produce_data_num = total_data * 50 * 49

        # data_num_per_file = int(produce_data_num / file_num)
        
        concat_dict = list()

        # data_num = 0
        # file_count = 1
        sample_num = 5
        for n, data in enumerate(tqdm(token_file)):
            for k in range(sample_num):
                temp_dict = dict()
                i = random.randint(0, 49)
                j = random.randint(0, 49)
                while(i == j):
                    i = random.randint(0, 49)
                    j = random.randint(0, 49)
                first_seq = data['token'][i]
                second_seq = data['token'][j]

            # for i, first_seq in enumerate(data['token']):
            #     for j, second_seq in enumerate(data['token']):
                if (i == j):
                    continue
                temp_dict['token'] = first_seq + second_seq[1:]
                if (i > j):
                    temp_dict['label'] = 0
                else:
                    temp_dict['label'] = 1
                    # data_num += 1
            
                    # if (data_num % data_num_per_file == 0):
                    #     if (not os.path.exists(f'/mnt/disk3/Alfred/Rescoring/data/aishell/train/token/concat/{s}')) :
                    #         os.makedirs(f'/mnt/disk3/Alfred/Rescoring/data/aishell/train/token/concat/{s}')
                    
                    #     with open(f'/mnt/disk3/Alfred/Rescoring/data/aishell/train/token/concat/{s}/token_concat_{file_count}.json', 'w') as fw:
                    #         json.dump(
                    #             concat_dict, fw, ensure_ascii = False, indent = 4
                    #         )
                    #     file_count += 1
                    #     data_num = 0
                    #     concat_dict = {
                    #         'token': [],
                    #         'label': []
                    #     }
                concat_dict.append(temp_dict)
                
    with open(f'/mnt/disk3/Alfred/Rescoring/data/aishell/train/token/concat/{s}/token_concat.json', 'w') as fw:
        json.dump(
            concat_dict, fw, ensure_ascii = False, indent = 4
        )
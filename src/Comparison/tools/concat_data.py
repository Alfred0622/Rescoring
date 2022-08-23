# concat the token sequence for comparison
import json
from tqdm import tqdm
import os
import random

random.seed(42)
nbest = 20
dataset = ['dev']
setting = ['noLM', 'withLM']

# train & valid
for s in setting:
    for task in dataset:
        if (nbest > 20):
            sample_num = 20
        else:
            sample_num = -1

        if (task == 'dev'):
            save_path = 'valid'
            sample_num = 2

        else: save_path = task
        print(f"file: /mnt/disk1/Alfred/Rescoring/data/aishell/{task}/token/token_{s}_50best.json")
        with open(f'/mnt/disk1/Alfred/Rescoring/data/aishell/{task}/token/token_{s}_50best.json') as f:
            token_file = json.load(f)
            total_data = len(token_file)
            print(f"total_data:{total_data}")
            
            concat_dict = list()


            if (sample_num > 0):
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

                        if (i == j):
                            continue
                        temp_dict['token'] = first_seq + second_seq[1:]
                        if (i > j):
                            temp_dict['label'] = 0
                        else:
                            temp_dict['label'] = 1
                        concat_dict.append(temp_dict)
            else:
                for n, data in enumerate(tqdm(token_file)): 
                    for i in range(nbest):
                        for j in range(nbest):
                            temp_dict = dict()
                            if (i == j):
                                continue
                            first_seq = data['token'][i]
                            second_seq = data['token'][j]

                            temp_dict['token'] = first_seq + second_seq[1:]
                            if (i < j): 
                                temp_dict['label'] = 1
                            else :
                                temp_dict['label'] = 0
                            concat_dict.append(temp_dict)


            if (not os.path.exists(f'../data/aishell/{save_path}/{s}')):
                os.makedirs(f"../data/aishell/{save_path}/{s}")
            print(f'nbest:{nbest}')
            print(f'total num should be:{total_data * nbest * (nbest - 1)}')    
            print(f'total_data_num : {len(concat_dict)}')
            with open(f'../data/aishell/{save_path}/{s}/token_concat.json', 'w') as fw:
                json.dump(
                    concat_dict, fw, ensure_ascii = False, indent = 4
                )

recog_set = ['dev', 'test']
# dev & test
for s in setting:
    for task in recog_set:
        
        print(f"file: /mnt/disk1/Alfred/Rescoring/data/aishell/{task}/token/token_{s}_50best.json")
        with open(
            f"/mnt/disk1/Alfred/Rescoring/data/aishell/{task}/token/token_{s}_50best.json"
        ) as f:
            load_data = json.load(f)
            save_list = list()
            for n, data in enumerate(load_data):
                temp_dict = dict()
                token_list = list()
                pair_list = list()
                temp_dict['name'] = data['name']
                temp_dict['ref'] = data['ref']
                temp_dict['text'] = data['text']
                temp_dict['score'] = data['score']
                temp_dict['err'] = data['err']

                for i in range(nbest):
                    for j in range(i, nbest):
                        first_seq = data['token'][i]
                        second_seq = data['token'][j]

                        token_list.append(first_seq + second_seq[1:])
                        
                        pair_list.append([i, j])
                
                temp_dict['token'] = token_list
                temp_dict['pair'] = pair_list
                save_list.append(temp_dict)
        print(f'total_data_num : {len(save_list)}')
        with open(
            f'../data/aishell/{task}/{s}/token.json', 'w'
        ) as fw:
            json.dump(save_list, fw, ensure_ascii = False, indent = 4)
        
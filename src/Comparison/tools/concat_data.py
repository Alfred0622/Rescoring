# concat the token sequence for comparison
import json
from tqdm import tqdm
from pathlib import Path
import os
import random
from transformers import BertTokenizer

random.seed(42)
nbest = 20
name = 'tedlium2'
dataset = ['train', 'dev']
setting = ['noLM', 'withLM']

if (name in ['tedlium2', 'librispeech']):
    tokenizer = BertTokenizer.from_pretrained(f'bert-base-uncased')
elif (name in ['aishell', 'aishell2']):
    tokenizer = BertTokenizer.from_pretrained(f'bert-base-chinese')
elif (name in ['csj']):
    pass

concat_train = True
# train & valid
if (concat_train):
    for s in setting:
        for task in dataset:
            if (nbest > 20):
                sample_num = 20
            else:
                sample_num = -1

            if (task == 'dev'):
                save_file = 'valid'
                sample_num = 2
            else: 
                save_file = task

            print(f"file: ../../../data/{name}/data/{s}/{task}/data.json")
            with open(f'../../../data/{name}/data/{s}/{task}/data.json') as f:
                token_file = json.load(f)
                total_data = len(token_file)
                print(f"total_data:{total_data}")

                concat_dict = list()

                if (sample_num > 0): # random choose
                    for n, data in enumerate(tqdm(token_file)):
                        for k in range(sample_num):
                            temp_dict = dict()
                            i = random.randint(0, len(data['hyp']) - 1)
                            j = random.randint(0, len(data['hyp']) - 1)
                            # print(len(data['hyp']))
                            # print(len(data['err']))
                            # print(f'i:{i}, j:{j}')
                            err_1 = data['err'][i]
                            err_2 = data['err'][j]

                            cer_1 = (
                                (err_1[1] + err_1[2] + err_1[3])/(err_1[0] + err_1[1] + err_1[2])
                            )
                            cer_2 = (
                                (err_2[1] + err_2[2] + err_2[3])/(err_2[0] + err_2[1] + err_2[2])
                            )

                            reshuffle = 0
                            while((i == j  or cer_1 == cer_2) and reshuffle < 50):
                                # print(f'reshuffle')
                                i = random.randint(0, len(data['hyp']) - 1)
                                j = random.randint(0, len(data['hyp']) - 1)
                                # print(len(data['hyp']))
                                # print(len(data['err']))
                                # print(f'i:{i}, j:{j}')
                                err_1 = data['err'][i]
                                err_2 = data['err'][j]

                                cer_1 = (
                                    (err_1[1] + err_1[2] + err_1[3])/(err_1[0] + err_1[1] + err_1[2])
                                )
                                cer_2 = (
                                    (err_2[1] + err_2[2] + err_2[3])/(err_2[0] + err_2[1] + err_2[2])
                                )
                                reshuffle += 1
                            
                            if (reshuffle >= 50):
                                continue
                            first_seq = data['hyp'][i]
                            second_seq = data['hyp'][j]

                            temp_dict = {
                                "hyp1" : first_seq,
                                "hyp2" : second_seq,
                                "label": 0
                            }
                            if (cer_1 > cer_2):
                                temp_dict['label'] = 0
                            else:
                                temp_dict['label'] = 1
                            concat_dict.append(temp_dict)
                else:
                    print(f'{nbest} best')
                    sample_num = nbest * (nbest - 1)
                    same_cer = 0
                    for n, data in enumerate(tqdm(token_file)):
                        if (nbest > len(data['hyp'])):
                            hyp_num = len(data['hyp'])
                        else:
                            hyp_num = nbest
                        for i in range(hyp_num):
                            for j in range(hyp_num):
                                temp_dict = dict()
                                if (i == j):
                                    continue
                                first_seq = data['hyp'][i]
                                second_seq = data['hyp'][j]
                                if (task in ['train', 'dev']):
                                    err_1 = data['err'][i]
                                    err_2 = data['err'][j]

                                    cer_1 = (
                                        (err_1[1] + err_1[2] + err_1[3])/(err_1[0] + err_1[1] + err_1[2])
                                    )
                                    cer_2 = (
                                        (err_2[1] + err_2[2] + err_2[3])/(err_2[0] + err_2[1] + err_2[2])
                                    )

                                    if (cer_1 == cer_2):
                                        same_cer += 1
                                        continue
                                concat_seq = first_seq + "[SEP]" + second_seq

                                temp_dict = {
                                    "hyp1": first_seq,
                                    "hyp2": second_seq,
                                    "label": 0
                                }
                                if (cer_1 < cer_2): 
                                    temp_dict['label'] = 1
                                else :
                                    temp_dict['label'] = 0
                                concat_dict.append(temp_dict)
                    print(f'same cer num:{same_cer}')


                if (not os.path.exists(f'../data/{name}/{save_file}/{s}/{nbest}best')):
                    os.makedirs(f"../data/{name}/{save_file}/{s}/{nbest}best")
                print(f'nbest:{nbest}')
                print(f'total num should be:{total_data * sample_num}')    
                
                print(f'total_data_num : {len(concat_dict)}')
                save_path = Path(f"../data/{name}/{save_file}/{s}/{nbest}best")
                save_path.mkdir(parents = True, exist_ok = True)
                with open(f'{save_path}/data.json', 'w') as fw:
                    json.dump(
                        concat_dict, fw, ensure_ascii = False, indent = 4
                    )

concat_test = True
if (concat_test):
    recog_set = ['dev', 'test']
    # dev & test
    for s in setting:
        for task in recog_set:
            print(f"file: ../../../data/{name}/data/{s}/{task}/data.json")
            with open(
                f"../../../data/{name}/data/{s}/{task}/data.json"
            ) as f:
                load_data = json.load(f)
                save_list = list()
                for n, data in enumerate(load_data):
                    temp_dict = dict()
                    temp_dict['name'] = data['name']
                    temp_dict['hyp'] = list()
                    temp_dict['ref'] = data['ref']
                    temp_dict['texts'] = list()
                    temp_dict['am_score'] = data['am_score']
                    temp_dict['ctc_score'] = data['ctc_score']
                    temp_dict['score'] = data['score']
                    temp_dict['err'] = data['err']
                    temp_dict['nbest'] = len(data['err'])
                    if ('lm_score' in data.keys()):
                        temp_dict['lm_score'] = data['lm_score']
                    
                    for seq in data['hyp']:
                        temp_dict['texts'].append(seq)


                    for i in range(nbest - 1):
                        for j in range(i + 1, nbest):
                            first_seq = data['hyp'][i]
                            second_seq = data['hyp'][j]

                            temp_dict['hyp'].append(
                                {
                                    "hyp1": first_seq,
                                    "hyp2": second_seq,
                                    "pair": [i, j]
                                }
                            )
                    save_list.append(temp_dict)
            print(f'total_data_num : {len(save_list)}')

            save_path = Path(f"../data/{name}/{task}/{s}/{nbest}best")
            save_path.mkdir(parents = True, exist_ok = True)
        
            with open(
                f'{save_path}/data.json', 'w'
            ) as fw:
                json.dump(save_list, fw, ensure_ascii = False, indent = 4)
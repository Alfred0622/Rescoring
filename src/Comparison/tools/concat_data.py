# concat the token sequence for comparison
import json
from tqdm import tqdm
from pathlib import Path
import os
import random
import torch
from transformers import BertTokenizer

random.seed(42)
nbest = 20

name = 'librispeech'
setting = ['noLM', 'withLM']

if (name in ['tedlium2', 'librispeech']):
    tokenizer = BertTokenizer.from_pretrained(f'bert-base-uncased')
elif (name in ['aishell', 'aishell2']):
    tokenizer = BertTokenizer.from_pretrained(f'bert-base-chinese')
elif (name in ['csj']):
    pass

concat_train = True
concat_test = True

# train & valid
if (concat_train):

    if (name in ['tedlium2']):
        dataset = ['train', 'dev_trim']
    elif (name in ['aishell2']):
        dataset = ["train", "dev_ios"]
    elif (name in ['librispeech']):
        dataset = ["train", 'dev_clean', 'dev_other']
    else:
        dataset = ['train', 'dev']
    for s in setting:
        for task in dataset:
            if (nbest > 20):
                sample_num = 20
            else:
                sample_num = -1

            if (task in ['dev', 'dev_trim', 'dev_ios', 'dev_clean', 'dev_other']):
                save_file = 'valid'
                concat_nbest = 5
            else: 
                save_file = task
                concat_nbest = nbest

            print(f"file: ../../../data/{name}/data/{s}/{task}/data.json")
            with open(f'../../../data/{name}/data/{s}/{task}/data.json') as f:
                token_file = json.load(f)
                total_data = len(token_file)
                print(f"total_data:{total_data}")

                if (name in ['librispeech'] and task in ['dev_other']):
                    with open(f"../data/{name}/valid/{s}/{concat_nbest}best/data.json") as f:
                        concat_dict = json.load(f)
                else:
                    concat_dict = list()

                if (sample_num > 0): # random choose
                    for n, data in enumerate(tqdm(token_file)):
                        for k in range(sample_num):
                            temp_dict = dict()
                            i = random.randint(0, len(data['hyps']) - 1)
                            j = random.randint(0, len(data['hyps']) - 1)
                            # print(len(data['hyps']))
                            # print(len(data['err']))
                            # print(f'i:{i}, j:{j}')
                            err_1 = data['err'][i]['err']
                            err_2 = data['err'][j]['err']

                            reshuffle = 0
                            while((i == j  or err_1 == err_2) and reshuffle < 50):
                                # print(f'reshuffle')
                                i = random.randint(0, len(data['hyps']) - 1)
                                j = random.randint(0, len(data['hyps']) - 1)
                                # print(len(data['hyps']))
                                # print(len(data['err']))
                                # print(f'i:{i}, j:{j}')
                                err_1 = data['err'][i]['err']
                                err_2 = data['err'][j]['err']

                                reshuffle += 1
                            
                            if (reshuffle >= 50):
                                continue
                            first_seq = data['hyps'][i]
                            second_seq = data['hyps'][j]

                            am_score = [data['am_score'][i], data['am_score'][j]]
                            ctc_score = [data['ctc_score'][i], data['ctc_score'][j]]
                            if (len(data['lm_score']) > 0):
                                lm_score = [data['lm_score'][i], data['lm_score'][j]]
                            else:
                                lm_score = [0.0, 0.0]

                            temp_dict = {
                                "name" : data['name'],
                                "hyp1" : first_seq,
                                "hyp2" : second_seq,
                                "am_score": am_score,
                                "ctc_score": ctc_score,
                                "lm_score": lm_score,
                                "label": 0
                            }
                            if (err_1 > err_2):
                                temp_dict['label'] = 0
                            else:
                                temp_dict['label'] = 1
                            concat_dict.append(temp_dict)
                else:
                    print(f'{concat_nbest} best')
                    sample_num = concat_nbest * (concat_nbest - 1)
                    same_cer = 0
                    for n, data in enumerate(tqdm(token_file, ncols = 100)):
                        if (concat_nbest > len(data['hyps'])):
                            hyp_num = len(data['hyps'])
                        else:
                            hyp_num = concat_nbest
                        for i in range(hyp_num):
                            for j in range(hyp_num):
                                if (i == j):
                                    continue
                                first_seq = data['hyps'][i]
                                second_seq = data['hyps'][j]
                                if (save_file in ['train', 'valid']):
                                    err_1 = data['err'][i]['err']
                                    err_2 = data['err'][j]['err']

                                    if (err_1 == err_2):
                                        same_cer += 1
                                        continue
                                
                                if ('am_score' in data.keys() and isinstance(data['am_score'], list) and len(data['am_score']) > 0):
                                    am_score = [data['am_score'][i], data['am_score'][j]]
                                else: 
                                    am_score = [0.0, 0.0]

                                if ('ctc_score' in data.keys() and isinstance(data['ctc_score'], list) and len(data['ctc_score']) > 0):
                                    ctc_score = [data['ctc_score'][i], data['ctc_score'][j]]
                                else:
                                    ctc_score = [0.0, 0.0]

                                if ('lm_score' in data.keys() and isinstance(data['lm_score'], list) and len(data['lm_score']) > 0):
                                    lm_score = [data['lm_score'][i], data['lm_score'][j]]
                                else:
                                    lm_score = [0.0, 0.0]

                                temp_dict = {
                                    "name" : data['name'],
                                    "hyp1" : first_seq,
                                    "hyp2" : second_seq,
                                    "am_score": am_score,
                                    "ctc_score": ctc_score,
                                    "lm_score": lm_score,
                                    "label": 0
                                }
                                if (err_1 < err_2): 
                                    temp_dict['label'] = 1
                                else :
                                    temp_dict['label'] = 0
                                concat_dict.append(temp_dict)

                                
                    print(f'same cer num:{same_cer}')


                if (not os.path.exists(f'../data/{name}/{save_file}/{s}/{concat_nbest}best')):
                    os.makedirs(f"../data/{name}/{save_file}/{s}/{concat_nbest}best")
                print(f'concat_nbest:{concat_nbest}')
                print(f'total num should be:{total_data * sample_num}')    
                
                print(f'total_data_num : {len(concat_dict)}')
                save_path = Path(f"../data/{name}/{save_file}/{s}/{concat_nbest}best")
                save_path.mkdir(parents = True, exist_ok = True)

                with open(f'{save_path}/data.json', 'w') as fw:
                    json.dump(
                        concat_dict, fw, ensure_ascii = False, indent = 4
                    )

if (concat_test):
    if (name in ['tedlium2']):
        recog_set = ['dev', 'dev_trim', 'test']
    elif (name in ["aishell2"]):
        recog_set = ['dev_ios', 'test_ios', 'test_mic', 'test_android']
    elif (name in ['librispeech']):
        recog_set = ['dev_clean', 'dev_other', 'test_clean', 'test_other']
    elif (name in ['csj']):
        recog_set = ['dev', 'eval1', 'eval2', 'eval3']
    else:
        recog_set = ['dev', 'test']
    # recog_set = ['dev', 'test']
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
                    temp_dict['hyps'] = list()
                    temp_dict['ref'] = data['ref']
                    temp_dict['texts'] = list()
                    if ('am_score' in data.keys()):
                        temp_dict['am_score'] = data['am_score']
                    if ('ctc_score' in data.keys()):
                        temp_dict['ctc_score'] = data['ctc_score']
                    if ('score' in data.keys()):
                        temp_dict['score'] = data['score']
                    temp_dict['err'] = data['err']
                    temp_dict['nbest'] = len(data['err'])

                    if ('lm_score' in data.keys()):
                        temp_dict['lm_score'] = data['lm_score']
                    
                    for seq in data['hyps']:
                        temp_dict['texts'].append(seq)
                    
                    real_nbest = nbest
                    if (len(data['hyps']) < nbest):
                        real_nbest = len(data['hyps'])

                    for i in range(real_nbest):
                        for j in range(real_nbest):
                            if (i == j): continue
                            first_seq = data['hyps'][i]
                            second_seq = data['hyps'][j]

                            if ('am_score' in data.keys() and isinstance(data['am_score'], list) and len(data['am_score']) > 0):
                                am_score = [data['am_score'][i], data['am_score'][j]]
                            else: 
                                am_score = [0.0, 0.0]

                            if ('ctc_score' in data.keys() and isinstance(data['ctc_score'], list) and len(data['ctc_score']) > 0):
                                ctc_score = [data['ctc_score'][i], data['ctc_score'][j]]
                            else:
                                ctc_score = [0.0, 0.0]

                            if ('lm_score' in data.keys() and isinstance(data['lm_score'], list) and len(data['lm_score']) > 0):
                                lm_score = [data['lm_score'][i], data['lm_score'][j]]
                            else:
                                lm_score = [0.0, 0.0]

                            pair_dict = {
                                "hyp1" : first_seq,
                                "hyp2" : second_seq,
                                "am_score": am_score,
                                "ctc_score": ctc_score,
                                "lm_score": lm_score,
                                "pair":[i, j]
                            }
                            temp_dict['hyps'].append(pair_dict)

                    save_list.append(temp_dict)
            print(f'total_data_num : {len(save_list)}')

            save_path = Path(f"../data/{name}/{task}/{s}/{nbest}best")
            save_path.mkdir(parents = True, exist_ok = True)
        
            with open(
                f'{save_path}/data.json', 'w'
            ) as fw:
                json.dump(save_list, fw, ensure_ascii = False, indent = 4)
    
if (name == 'librispeech'):
    with open(f"../data/{name}/dev_clean/{s}/5best/data.json") as clean , \
            open(f"../data/{name}/dev_other/{s}/5best/data.json") as other :
        
        valid_json = json.load(clean)
        valid_json = valid_json + json.load(other)

        with open(f"../data/{name}/valid/{s}/{nbest}best/data.json", 'w') as valid:
            json.dump(valid_json, valid, ensure_ascii = False, indent = 1)
import json
import sys
from pathlib import Path

dataset = sys.argv[1]
# target_path = sys.argv[2]

setting = ['noLM', 'withLM']

if (dataset in ['aishell', 'tedlium2']):
    data_task = ['train', 'dev', 'test']
elif (dataset in ['csj']):
    data_task = ['train', 'dev', 'eval1', 'eval2', 'eval3']
elif (dataset in ['librispeech']):
    data_task = ['train', 'dev_other','dev_clean', 'test_other', 'test_clean']
elif (dataset in ['aishell2']):
    data_task = ['train', 'dev', 'test_mic', 'test_ios', 'test_android']

if (dataset in ['tedlium2', 'librispeech']):
    trn_name = ".wrd.trn"
else:
    trn_name = '.trn'


for task in data_task:
    print(f'{task}')
    for s in setting:
        print(f'----{s}')
        data_list = list()
        
        nbest_count = 0
        nbest_hyp = []
        single_set = []
        set_num = 0

        nbest_num = list()

        with open(f"../data/{dataset}/raw/{s}/{task}/data.json") as f, \
             open(f"../data/{dataset}/raw/{s}/{task}/hyp{trn_name}") as hf:
            data_json = json.load(f)
            
            for key in data_json['utts'].keys():
                nbest_num.append(len(data_json['utts'][key]['output']))
            
            print(len(nbest_num))

            for i, line in enumerate(hf):
                parenthesis_index = line.find('(')
                single_set.append(line[:parenthesis_index].strip())
                nbest_count += 1
                if (nbest_count % nbest_num[set_num] == 0):
                    nbest_hyp.append(single_set)
                    single_set = []
                    nbest_count = 0
                    set_num += 1

        print(len(nbest_hyp))

        for key, hyps in zip(data_json['utts'].keys(), nbest_hyp):
            data_dict = dict()
            data_dict['name'] = key
            data_dict['hyps'] = list()
            data_dict['hyps_id'] = list()
            data_dict['ref'] = data_json['utts'][key]['output'][0]['text']
            data_dict['ref_id'] = data_json['utts'][key]['output'][0]['tokenid']
    
            data_dict['score'] = list()
            data_dict['am_score'] = list()
            data_dict['ctc_score'] = list()
            data_dict['lm_score'] = list()

            for nbest in data_json['utts'][key]['output']:
                # split_char = nbest['rec_token'].strip().split(' ')[:-1]
                split_id = nbest['rec_tokenid'].strip().split(' ')[:-1]

                # data_dict['hyps'].append(split_char)
                data_dict['hyps_id'].append(split_id)

                data_dict['score'].append(nbest['score'])
                if ('am_score' in nbest.keys()):
                    data_dict['am_score'].append(nbest['am_score'])
                if ('ctc_score' in nbest.keys()):
                    data_dict['ctc_score'].append(nbest['ctc_score'])
                if ('lm_score' in nbest.keys()):
                    data_dict['lm_score'].append(nbest['lm_score'])
            data_dict['hyps'] = hyps.copy()
            data_list.append(data_dict)
        
        token_path = Path(f"../data/{dataset}/token/{s}/{task}")
        token_path.mkdir(parents = True, exist_ok = True)
        with open(f"{token_path}/token.json", 'w') as f:
            json.dump(data_list, f, ensure_ascii = False, indent = 4)


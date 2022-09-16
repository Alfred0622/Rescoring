import json
import os
# get training data for correction

setting = ['noLM','withLM']
task = ['train', 'dev', 'valid','test']
nbest = 20
dataset = 'aishell'

for s in setting:
    for t in task:
        print(f'{s}:{t}')
        name = t
        topk = nbest

        if (t == 'valid'):
            name = 'dev'

        if (t in ['valid', 'dev', 'test']):
            topk = 1
            
        
        with open(f'./data/{dataset}/{s}/{name}/token.json', 'r') as f:
            data = json.load(f)
            new_data = {
                'token':list(),
                'ref_token':list(),
                'ref':list()
            }
            for d in data:
                token = d['token'][:topk]
                err = d['err'][:topk]
                ref = d['ref_text']
                ref_token = d['ref_token']
            
                for token_unit, err_unit in zip(token, err):
                    cer = (
                        (err_unit[1] + err_unit[2] + err_unit[3]) / (err_unit[0] + err_unit[1] + err_unit[2])
                    )

                    if (cer <= 0.5):
                        new_data['token'].append(token_unit)
                        new_data['ref_token'].append(ref_token)
                        new_data['ref'].append(ref)
            
            if (not os.path.exists(f'./data/{dataset}/{s}/{t}/{topk}best')):
                os.makedirs(f'./data/{dataset}/{s}/{t}/{topk}best')
            
            with open(f'./data/{dataset}/{s}/{t}/{topk}best/token.json', 'w') as fw:
                json.dump(new_data, fw, ensure_ascii=False, indent = 4)
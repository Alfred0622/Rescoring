import os
import sys
import json
from transformers import BertTokenizer
#Usage: python ./gen_plain.py <dataset> <nbest> <delimeter>

setting = ['withLM', 'noLM']
task_group = ['train', 'dev', 'test']

print(len(sys.argv))
assert len(sys.argv) == 4, "Usage: python ./gen_plain.py <dataset> <nbest> <delimeter>"

tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')

dataset = sys.argv[1]
nbest = int(sys.argv[2])
delimeter = sys.argv[3]  # delimeter : '#'

if (delimeter == 'SEP'):
    delimeter = '[SEP]'
delimeter_token = tokenizer.convert_tokens_to_ids(delimeter)
print(f'delimeter:{delimeter} = {delimeter_token}')

for s in setting:
    for task in task_group:
        print(f'{s}:{task}')
        with open(f'../../../data/{dataset}/{task}/data/data_{s}.json') as f:
            data = json.load(f)

            gen_data = list()
            for d in data:
                name = d['name']
                tokens = d['token']
                ref_text = d['ref']

                concat_token = str()

                for i, t in enumerate(tokens[:nbest]):
                    
                    if (i == 0): # [CLS] A B C [SEP] -> [CLS] A B C #
                        concat_token += t # first adding
                    else:
                        concat_token = concat_token + delimeter + t # [CLS] A B C [SEP] -> [CLS] A B C #
                
                gen_data.append(
                    {
                        'name':name,
                        'token':concat_token,
                        'ref': ref_text,
                    }
                )
        if (not os.path.exists(f"../data/{dataset}/{s}/{t}/{nbest}plain/")):
            os.makedirs(f'../data/{dataset}/{s}/{t}/{nbest}plain')

        with open(f'../data/{dataset}/{s}/{t}/{nbest}plain/data.json', 'w') as fw:
            json.dump(gen_data, fw, ensure_ascii=False, indent = 4)

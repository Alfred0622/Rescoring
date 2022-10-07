import json

compare = ['train', 'dev', 'test']

for task in compare:
    print(f'task:{task}')
    with open(f'./data/aishell/data/noLM/{task}/data.json') as f,\
        open(f'./data/old_aishell/data/{task}/data.json') as old_f:
         
         json_data = json.load(f)
         old_json = json.load(old_f)

         new_compare = []
         old_compare = []

         for data in json_data:
            new_compare.append(
                {
                    "hyp": data['hyp'][:10],
                    "ref": data['ref']
                }
            )
         for data in old_json:
            old_compare.append(
                {
                    "hyp": data['hyp'],
                    "ref": data['ref']
                }
            )
        
    with open(f'./data/compare_aishell/data_{task}.json', 'w') as f,\
        open(f'./data/compare_aishell/old_data_{task}.json', 'w') as old_f:
        json.dump(new_compare, f, ensure_ascii = False, indent = 4)
        json.dump(old_compare, old_f, ensure_ascii = False, indent = 4)
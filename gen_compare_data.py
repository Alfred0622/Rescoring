import json

compare = ['train', 'dev', 'test']

for task in compare:
    print(f'task:{task}')
    with open(f'./data/aishell/data/noLM/{task}/data.json') as f,\
         open(f'./data/aishell/data/withLM/{task}/data.json') as wf, \
         open(f'./data/old_aishell/data/{task}/data.json') as old_f:
         
         json_noLM_data = json.load(f)
         json_withLM_data = json.load(wf)
         old_json = json.load(old_f)

         new_noLM_compare = []
         new_withLM_compare = []
         old_compare = []

         for data in json_noLM_data:
            new_noLM_compare.append(
                {
                    "hyp": data['hyp'][:10],
                    "ref": data['ref']
                }
            )
         for data in json_withLM_data:
            new_withLM_compare.append(
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
        
    with open(f'./data/compare_aishell/data_{task}_noLM.json', 'w') as f,\
        open(f'./data/compare_aishell/data_{task}_withLM.json', 'w') as wf,\
        open(f'./data/compare_aishell/old_data_{task}.json', 'w') as old_f:
        json.dump(new_noLM_compare, f, ensure_ascii = False, indent = 4)
        json.dump(new_withLM_compare, wf, ensure_ascii = False, indent = 4)
        json.dump(old_compare, old_f, ensure_ascii = False, indent = 4)
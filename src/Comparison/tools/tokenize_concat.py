import json

dataset = 'tedlium2'
setting = ['noLM', 'withLM']

if (dataset in {"csj"}):
    data_name = ['dev', 'eval1', 'eval2', 'eval3']
else:
    data_name = ['dev', 'test']

for s in setting:
    for name in data_name:
        print(f"{s}:{name}")

        file_name = f"../data/{dataset}/{name}/{s}"
        with open(file_name) as f:
            data_json = json.load(f)
        
        for data in data_json:
            pass
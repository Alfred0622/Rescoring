import json
import os
import subprocess
from tqdm import tqdm
from pathlib import Path

os.environ["PATH"] = f"/mnt/disk6/Alfred/Rescoring/src/Correction/jumanpp-2.0.0-rc3/bld/bin:{os.environ['PATH']}"
tokenization = False
concat = True

task = ['withLM']
datasets = [f"train_{i}" for i in range(1,17)]
# datasets = ['dev', 'eval1', 'eval2', 'eval3']

if (tokenization):
    for s in task:
        for dataset in datasets:
            print(f"{s}: {dataset}")
            data_file = f"/mnt/disk6/Alfred/Rescoring/data/csj/data/{s}/{dataset}/data.json"
            dest_file = f"/mnt/disk6/Alfred/Rescoring/src/Correction/data/csj/{s}/{dataset}"
            data_list = []
            with open(data_file) as f:
                data_json = json.load(f)
                for i, data in tqdm(enumerate(data_json), ncols = 100, total= len(data_json)):
                    temp_dict = {}
                    temp_dict['name'] = data['name']
                    temp_list = []
                    for j, hyp in enumerate(data['hyps'][:10]):

                        hyp = "".join(hyp.split()).replace("<eos>", "")
                        # print(f"hyp:{hyp}")
                        command = "echo " + hyp + " | jumanpp --segment"
                        result = subprocess.check_output(f"echo {hyp} | jumanpp --segment", shell = True).decode('utf-8')
                        result = result.replace("\n", "")
                        assert("\n" not in result), "ERROR"
                        temp_list.append(result)
                        # print(type(result))
                        # print(f'result:{result}')

                    ref = "".join(data['ref'].split()).replace("<eos>", "")
                    result = subprocess.check_output(f"echo {ref} | jumanpp --segment", shell = True).decode('utf-8')
                    result = result.replace("\n", "")
                    assert("\n" not in result), "ERROR"
                    temp_dict['hyps'] = temp_list
                    temp_dict['ref'] = result
                    data_list.append(temp_dict)
            
            dest_file = Path(dest_file)
            dest_file.mkdir(parents=True, exist_ok=True)

            with open(f"{dest_file}/data.json", 'w') as d:
                json.dump(data_list, d, ensure_ascii=False, indent = 1)

if (concat):
    final_json = []
    for s in task:
        for dataset in datasets:
            print(f"concat {s}: {dataset}")
            data_file = f"../data/csj/{s}/{dataset}/data.json"
            with open (data_file) as f:
                data_json = json.load(f)
                final_json += data_json
        
        with open(f"../data/csj/{s}/train/data.json", 'w') as d:
            json.dump(final_json, d, ensure_ascii = False, indent = 1)
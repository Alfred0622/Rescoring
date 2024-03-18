import json
import sys
from pathlib import Path

dataset_name = 'librispeech'

to_combine = ['dev_clean', 'dev_other']
methods = ['CLM', 'MLM', 'RescoreBert_MD', 'RescoreBert_MWER', 'RescoreBert_MWED', 'BertAlsem', "Bert_sem"]

for method in methods:
    save_list = []
    for task in to_combine:
        file_name = f"/mnt/disk6/Alfred/Rescoring/data/result/librispeech/withLM/{task}/10best/{method}/data.json"

        with open(file_name) as f:
            data_json = json.load(f)
        save_list += data_json
    
    print(f'{method}: len of data:{len(save_list)}')
    save_path = Path(f"/mnt/disk6/Alfred/Rescoring/data/result/librispeech/withLM/valid/10best/{method}")
    save_path.mkdir(exist_ok = True, parents = True)

    with open(f"{save_path}/data.json", "w") as fp:
        json.dump(save_list, fp, ensure_ascii=False, indent=1)
        

import torch
import json
from src_utils.LoadConfig import load_config
from model.combineLinear import combineLinear
from pathlib import Path

# Step 1: Load datasets
# Load every data.json at each task, making 

data_path = Path("/mnt/disk6/Alfred/Rescoring/data/result/aishell/noLM/train/10best")

methods = ['CLM', 'MLM','RescoreBert_MD', 'RescoreBert_MWER', 'RescoreBert_MWED', 'Bert_sem', 'Bert_alsem', 'PBert', 'PBert_LSTM']
score_dict = {}
for method in methods:
    ensemble_path = f"{data_path}/{method}/data.json"

    with open(ensemble_path) as f:
        data_json = json.load(f)
    
    # take the score part in data_json
    # rescore_list = []
    # for data in data_json:
    #   rescore_list.append(data["rescore"])
    # score_dict[method] = rescore_list

# concat making a dataset
# ensemble_dataset = prepare_ensemble_dataset(score_dict)
# ensemble_loader = Dataloader(ensemble_dataset)

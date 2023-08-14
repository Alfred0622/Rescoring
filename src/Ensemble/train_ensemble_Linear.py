import torch
import json
import sys
sys.path.append("../")
from src_utils.LoadConfig import load_config
from model.combineLinear import combineLinear
from pathlib import Path
from torch.utils.data import DataLoader
from utils.Datasets import prepare_ensemble_dataset

config = "/mnt/disk6/Alfred/Rescoring/src/Ensemble/config/ensemble.yaml"
args, train_args, recog_args = load_config(config)
setting = 'withLM' if args['withLM'] else 'noLM'

# Step 1: Load datasets
# Load every data.json at each task, making 

data_path = Path("/mnt/disk6/Alfred/Rescoring/data/result/aishell/noLM/train/10best")

methods = ['CLM', 'MLM','RescoreBert_MD', 'RescoreBert_MWER', 'RescoreBert_MWED', 'Bert_sem', 'Bert_alsem', 'PBert', 'PBert_LSTM']
score_dict = {}
with open(f"/mnt/disk6/Alfred/Rescoring/data/{args['dataset']}/data/{setting}/train/data.json") as f:
    pass
for method in methods:
    ensemble_path = f"{data_path}/{method}/data.json"

    with open(ensemble_path) as f:
        data_json = json.load(f)
    
    score_dict = prepare_ensemble_dataset(data_json,score_dict)
    
    # take the score part in data_json
# concat making a dataset
ensemble_dataset = prepare_ensemble_dataset(score_dict)
ensemble_loader = DataLoader(ensemble_dataset)

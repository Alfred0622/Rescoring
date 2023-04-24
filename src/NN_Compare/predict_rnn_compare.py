import os
import torch
import sys
sys.path.append(f'..')
import logging
import json
from torch.data.utils import DataLoader
from jiwer import wer
from pathlib import Path
from src_utils import load_config, get_dict
from utils.load_config import load_config
from src_utils.get_recog_set import get_recog_set
from utils.Datasets import getRecogDataset
from model.utils.CollateFunc import recogBatch
from model.RNN_Rerank import RNN_Reranker

config_name = sys.argv[1]
args, train_args, recog_args = load_config(config_name)
setting = 'withLM' if args['withLM'] else 'noLM'

vocab_dict = f"./data/{args['dataset']}/lang1char/vocab.txt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(vocab_dict, 'r') as f:
    dict_file = f.read()

vocab_dict = get_dict(dict_file)

model = RNN_Reranker(
    vocab_size = len(vocab_dict),
    hidden_dim = 100,
    num_layers = 1,
    output_dim = 100, 
    dropout = 0.1,
    add_additional_feat = args['add_additional_feat'],
    add_am_lm_score = args['add_am_lm_score'],
    device = device
)
checkpoint_path = Path(f"./checkpoint/{args['dataset']}/{setting}")

checkpoint = torch.load(f"{checkpoint_path}/checkpoint_train_best.pt")
print(f"using epoch:{checkpoint['epoch']}")
model.load_state_dict(checkpoint['state_dict'])

recog_set = get_recog_set(args['dataset'])

for task in recog_set:
    with open(f"../../data/{args['dataset']}/token/{setting}/{task}/token.json") as f:
        data_json = json.load(f)
    
    dataset = getRecogDataset(data_json)

    dataloader = DataLoader(
        dataset = dataset,
        batch_size = recog_args['batch_size'],
        collate_fn = recogBatch,
    )

    result_dict = []
    result_hyp = []
    refs = []

    for data in dataloader:
        inputs = data['inputs'].to(device)

        am_score = data['am_score'].to(device)
        lm_score = data['lm_score'].to(device)
        ctc_score = data['ctc_score'].to(device)

        oracle = inputs[0]
        best_id = 0

        for i, hyp in enumerate(inputs[1:]):
            logit = model(
                oracle,
                hyp,
                am_score,
                lm_score,
                ctc_score
            ).logit
            if (torch.argmax(logit) == 1):
                oracle = hyp.clone()
                best_id = i + 1
        
        result_hyp.append(data['hyps'][best_id])
        refs.append(data['ref'])
        
        result_dict.append(
            {
                'best_id': best_id,
                'hyp':data['hyps'][best_id],
                "ref":data['ref']
            }
        )
    
    print(f'{task}: {wer(refs, result_hyp)}')
    with open(f"./data/{args['dataset']}/{setting}/{task}/{args['nbest']}best_resore_data.json", 'w') as f:
        json.dump(result_dict, f, ensure_ascii = False)
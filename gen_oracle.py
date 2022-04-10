import os
from tqdm import tqdm
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader

dev = "./data/aishell_dev"
test = "./data/aishell_test" 

file_name = [dev, test]

for n in file_name:
    print(f'processing: {n}')
    hyps = []
    refs = []
    with open(f'{n}/dataset.json', 'r') as f:
        data = json.load(f)
        for d in data:
            token = d['token']
            err = d['err']
            ref = d['ref']
            
            min_err = 1e8
            best_hyp = None
            for t, e in zip(token, err):
                cer = (e[1] + e[2] + e[3]) / (e[0] + e[1] + e[2])
                if (cer < min_err):
                    min_err = cer
                    best_hyp = t
            best_hyp = best_hyp[5:-5]
            best_hyp = " ".join([h for h in best_hyp])
            ref = ref[5:-5]
            ref = " ".join([r for r in ref])

            hyps.append(best_hyp)
            refs.append(ref)
    
    with open(f'{n}/oracle/hyp.trn', 'w') as h, open(f'{n}/oracle/ref.trn', 'w') as r:
        for i, temp in enumerate(zip(hyps, refs)):
            hyp, ref = temp
            h.write(f'{hyp} (oracle_{i + 1})\n')
            r.write(f'{ref} (oracle_{i + 1})\n')
            


    
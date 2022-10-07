import os
from tqdm import tqdm
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from jiwer import wer

choose_nbest = [50]

dataset = 'old_aishell'
file_name = ['dev', 'test']
setting = [""]

for best in choose_nbest:
    print(f"{best} best: ")
    for s in setting:
        for n in file_name:
            print(f"processing: {n}: {s}")
            top_1_hyp = []
            hyps = []
            refs = []
            total_c = 0
            total_s = 0
            total_d = 0
            total_i = 0
            with open(f"./data/{dataset}/data/{s}/{n}/data.json", "r") as f:
                data = json.load(f)
                for d in data:
                    token = d["hyp"]
                    err = d["err"]
                    ref = d["ref"]

                    min_err = 1e8
                    best_hyp = None
                    min_c = 0
                    min_s = 0
                    min_d = 0
                    min_i = 0

                    for t, e in zip(token, err):
                        cer = (e[1] + e[2] + e[3]) / (e[0] + e[1] + e[2])
                        if cer < min_err:
                            min_err = cer
                            best_hyp = t
                            min_c = e[0]
                            min_s = e[1]
                            min_d = e[2]
                            min_i = e[3]

                    total_c += min_c
                    total_d += min_d
                    total_i += min_i
                    total_s += min_s

                    top_1_hyp.append(token[0])
                    hyps.append(best_hyp)
                    refs.append(ref)

            # if not os.path.exists(f"./data/{dataset}/oracle/{s}/{n}"):
            #     os.makedirs(f"./data/{dataset}/oracle/{s}/{n}")

            # with open(f"./data/{dataset}/oracle/{s}/{n}/hyp_05.trn", "w") as h, open(
            #     f"./data/{dataset}/oracle/{s}/{n}/ref_05.trn", "w"
            # ) as r:
            #     for i, temp in enumerate(zip(hyps, refs)):
            #         hyp, ref = temp
            #         h.write(f"{hyp} (oracle_{i + 1})\n")
            #         r.write(f"{ref} (oracle_{i + 1})\n")
            cer = (total_i + total_s + total_d) / (total_c + total_s + total_d)
            print(f"{s} : {n} -- {round(cer, 5)}")
            print(f'correct:{total_c}')
            print(f'substitution:{total_s}')
            print(f'deletion:{total_d}')
            print(f'insert:{total_i}')
            
            print(f'baseline:{wer(refs, top_1_hyp)}')
            print(f'jiwer:{wer(refs, hyps)}')

            print('\n')

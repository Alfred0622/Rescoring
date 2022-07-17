import os
from tqdm import tqdm
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader

nbest = 50
dev = [f"./data/csj/dev"]
test = [f"./data/csj/eval1", f"./data/csj/eval2", f"./data/csj/eval3"]

choose_nbest = [10, 20, 30, 50]

file_name = dev + test
setting = ["noLM"]

for best in choose_nbest:
    print(f"{best} best: ")
    for s in setting:
        for n in file_name:
            print(f"processing: {n}: {s}")
            hyps = []
            refs = []
            total_c = 0
            total_s = 0
            total_d = 0
            total_i = 0
            with open(f"{n}/data_{s}.json", "r") as f:
                data = json.load(f)
                for d in data:
                    token = d["token"][:best]
                    err = d["err"][:best]
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
                    best_hyp = best_hyp
                    best_hyp = " ".join([h for h in best_hyp])
                    ref = ref
                    ref = " ".join([r for r in ref])

                    total_c += min_c
                    total_d += min_d
                    total_i += min_i
                    total_s += min_s

                    hyps.append(best_hyp)
                    refs.append(ref)

            if not os.path.exists(f"{n}/oracle"):
                os.mkdir(f"{n}/oracle")

            with open(f"{n}/oracle/{s}_{best}best_hyp.trn", "w") as h, open(
                f"{n}/oracle/{s}_{best}best_ref.trn", "w"
            ) as r:
                for i, temp in enumerate(zip(hyps, refs)):
                    hyp, ref = temp
                    h.write(f"{hyp} (oracle_{i + 1})\n")
                    r.write(f"{ref} (oracle_{i + 1})\n")
            cer = (total_i + total_s + total_d) / (total_c + total_s + total_d)
            print(f"{n}:{cer}")

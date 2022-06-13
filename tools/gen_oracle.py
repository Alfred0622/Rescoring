import os
from tqdm import tqdm
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader

nbest = 50
dev = f"./data/aishell_dev/{nbest}_best"
test = f"./data/aishell_test/{nbest}_best"

choose_nbest = [10, 20, 30]

file_name = [dev, test]

for best in choose_nbest:
    print(f"{best} best: ")
    for n in file_name:
        print(f"processing: {n}")
        hyps = []
        refs = []
        with open(f"{n}/dataset.json", "r") as f:
            data = json.load(f)
            for d in data:
                token = d["token"][:best]
                err = d["err"][:best]
                ref = d["ref"]

                min_err = 1e8
                best_hyp = None
                for t, e in zip(token, err):
                    cer = (e[1] + e[2] + e[3]) / (e[0] + e[1] + e[2])
                    if cer < min_err:
                        min_err = cer
                        best_hyp = t
                best_hyp = best_hyp[5:-5]
                best_hyp = " ".join([h for h in best_hyp])
                ref = ref[5:-5]
                ref = " ".join([r for r in ref])

                hyps.append(best_hyp)
                refs.append(ref)

        if not os.path.exists(f"{n}/oracle"):
            os.mkdir(f"{n}/oracle")

        with open(f"{n}/oracle/{best}best_hyp.trn", "w") as h, open(
            f"{n}/oracle/{best}best_ref.trn", "w"
        ) as r:
            for i, temp in enumerate(zip(hyps, refs)):
                hyp, ref = temp
                h.write(f"{hyp} (oracle_{i + 1})\n")
                r.write(f"{ref} (oracle_{i + 1})\n")

from re import L
from nBestAligner.nBestAlign import align, alignNbest
import json

dataset = ["train", "dev", "test"]

if __name__ == "__main__":
    nBest = 3
    for task in dataset:
        print(task)
        with open(f"./data/aishell_{task}/bart_token/token.json") as f:
            data = json.load(f)

        result_dict = []
        for d in data:
            temp_dict = dict()
            align_pair = align(d["token"], nBest=nBest, placeholder=0)
            align_result = alignNbest(align_pair, placeholder=0)

            temp_dict["token"] = align_result
            temp_dict["score"] = d["score"][:nBest]
            temp_dict['ref'] = d['ref']
            temp_dict["ref_token"] = d["ref_token"]
            temp_dict["err"] = d["err"][:nBest]
            result_dict.append(temp_dict)

        with open(f"./data/aishell_{task}/bart_token/align_token.json", "w") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)

import sys
import os
from models.nBestAligner.nBestAlign import align, alignNbest
import json
from transformers import BertTokenizer

dataset = ["train", "dev", "test"]
setting = ['noLM', 'withLM']
topk = 4


tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
placeholder = "*"
print(f"placeholder {placeholder} = {tokenizer.convert_tokens_to_ids(placeholder)}")
if __name__ == "__main__":
    for s in setting:
        for task in dataset:
            print(task)
            with open(f"../data/aishell/{s}/{task}/token.json") as f:
                data = json.load(f)

            result_dict = []
            for d in data:
                temp_dict = dict()
                align_pair = align(
                    d["token"],
                    nBest=topk,
                    placeholder=tokenizer.convert_tokens_to_ids(placeholder),
                )
                align_result = alignNbest(
                    align_pair, placeholder=tokenizer.convert_tokens_to_ids(placeholder)
                )

                temp_dict["token"] = align_result
                temp_dict["score"] = d["score"][:topk]
                temp_dict["ref"] = d["ref"]
                temp_dict["ref_token"] = d["ref_token"]
                temp_dict["err"] = d["err"][:topk]
                result_dict.append(temp_dict)

            with open(
                f"../data/aishell/{s}/{task}/{topk}_align_token.json", "w"
            ) as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)

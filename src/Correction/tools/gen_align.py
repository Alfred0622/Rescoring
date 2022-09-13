import sys
import os
from models.nBestAligner.nBestAlign import align, alignNbest
import json
from transformers import BertTokenizer

dataset = ["train", "dev", "test"]
setting = ['noLM', 'withLM']
topk = 3

use_concat = True


tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
placeholder = "-"
concat_ = '[SEP]'
print(f"placeholder {placeholder} = {tokenizer.convert_tokens_to_ids(placeholder)}")
print(f"concat with {concat_} = {tokenizer.convert_tokens_to_ids(concat_)}")
if __name__ == "__main__":
    for s in setting:
        for task in dataset:
            print(task)
            with open(f"../../data/aishell/data/{task}/data_{s}.json") as f:
                data = json.load(f)

            result_dict = []
            for d in data:
                temp_dict = dict()
                hyps = list()
                for t in d['token'][:topk]:
                    hyps.append([x for x in t])
                align_pair = align(
                    hyps,
                    nBest=topk,
                    placeholder=tokenizer.convert_tokens_to_ids(placeholder),
                )
                align_result = alignNbest(
                    align_pair, placeholder=tokenizer.convert_tokens_to_ids(placeholder)
                )

                if (use_concat):
                    to_concat = align_result.copy()
                    align_result = []
                    for i, align_unit in to_concat:
                        if (i == 0):
                            align_result = align_result + align_unit[:-1]
                        elif (i == len(to_concat)):
                            align_result = align_result + align_unit[1:]
                        else:
                            align_result = align_result + [concat_] + align_unit[1:]

                temp_dict["token"] = align_result
                temp_dict["score"] = d["score"][:topk]
                temp_dict["ref"] = d["ref"]
                temp_dict["ref_token"] = d["ref_token"]
                temp_dict["err"] = d["err"][:topk]
                result_dict.append(temp_dict)
            
            if (use_concat):
                name = 'align_concat'
            else:
                name = 'align'
            with open(
                f"../data/aishell/{s}/{task}/{topk}_{name}_token.json", "w"
            ) as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)

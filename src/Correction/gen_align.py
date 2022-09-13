import sys
import os
from models.nBestAligner.nBestAlign import align, alignNbest
import json
from transformers import BertTokenizer

dataset = ["train", "dev", "test"]
setting = ['noLM', 'withLM']
topk = 3


tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
placeholder = "-"
concat_token = '[SEP]'
use_concat = True

print(f"placeholder {placeholder} = {tokenizer.convert_tokens_to_ids(placeholder)}")
print(f"concat token {concat_token} = {tokenizer.convert_tokens_to_ids(concat_token)}")
print(f'Concat: {use_concat}')
concat_id = tokenizer.convert_tokens_to_ids(concat_token)
if __name__ == "__main__":
    for s in setting:
        for task in dataset:
            print(task)

            with open(f"../../data/aishell/{task}/data/data_{s}.json") as f:
                data = json.load(f)

            # result_dict = []
            result_dict = {
                "token" : list(),
                "ref": list(),
                "ref_token": list()
            }
            for d in data:
                temp_dict = dict()
                token_list = []
                for token in d["token"]:
                    temp_tokens = [t for t in token]
                    token_list.append(temp_tokens)

                align_pair = align(
                    token_list,
                    nBest=topk,
                    placeholder=placeholder,
                )

                align_result = alignNbest(
                    align_pair, placeholder=placeholder
                )

                if (use_concat):
                    align_str = list()
                    for i, alignment in enumerate(align_result):
                        if (i == 0):
                            align_str = alignment + [concat_token]
                        elif (i == len(align_result) - 1):
                            align_str = align_str + alignment
                        else:                            
                            align_str = align_str + alignment + [concat_token]
                else: 
                    align_str = align_result
                            

                result_dict["token"].append(align_str)
                result_dict["ref"].append(d["ref"])
            
            if (use_concat):
                token_type = f'{topk}align_concat'
            else:
                token_type = f'{topk}align'

            if (not os.path.exists(f"./data/aishell/{s}/{task}/{token_type}")):
                os.makedirs(f"./data/aishell/{s}/{task}/{token_type}")

            with open(
                f"./data/aishell/{s}/{task}/{token_type}/data.json", "w"
            ) as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)

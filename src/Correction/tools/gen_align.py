import sys
import os
sys.path.append("..")
from models.nBestAligner.nBestAlign import align, alignNbest
import json
from transformers import BertTokenizer, BartTokenizer
from pathlib import Path

dataset = sys.argv[1]


if (dataset in ['aishell', 'tedlium2', 'aishell_nbest']):
    split_set = ["train", "dev", "test"]
elif (dataset in ['aishell2']):
    split_set = ["train", "dev_ios", "test_ios", "test_android", "test_mic"]

setting = ['noLM', 'withLM']
topk = 4

use_concat = False


if (dataset in ['aishell', 'aishell2', 'aishell_nbest']):
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
elif (dataset in ['tedlium2', 'librispeech']):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

placeholder = "-"
concat_ = '[SEP]'
print(f"placeholder {placeholder} = {tokenizer.convert_tokens_to_ids(placeholder)}")
print(f"concat with {concat_} = {tokenizer.convert_tokens_to_ids(concat_)}")

if __name__ == "__main__":
    for s in setting:
        for task in split_set:
            print(f"task:{task}")
            if (dataset == 'aishell_nbest'):
                path = f"../../../data/aishell/4Best_aishell_{task}.json"
            else:
                path = f"../../../data/{dataset}/data/{s}/{task}/data.json"
            with open(path) as f:
                data = json.load(f)

            result_dict = []
            for i, d in enumerate(data):
                temp_dict = dict()
                hyps = list() 
                if (dataset in ['aishell_nbest', 'tedlium2']):
                    hyps = [tokenizer.tokenize(hyp) for hyp in d['hyps'][:topk]]
                else:
                    for t in d['hyps'][:topk]:
                        if (dataset in ['aishell', 'aishell2']):
                            hyps.append([x for x in t.replace("<space>", "space").split()])
                        elif (dataset in ['tedlium2', 'librispeech2']):
                            hyps.append(tokenizer.tokenize(t))
                align_pair = align(
                    hyps,
                    nBest=topk,
                    placeholder=placeholder,
                )
                if (task in ['train']):
                    if (len(align_pair) == 0): continue
                else:
                    assert(len(align_pair) > 0), f"Data:{i}, Hyps:{d['hyps'][:topk]}, align_pair:{align_pair}"

                align_result = alignNbest(
                    align_pair, placeholder=placeholder
                )

                for al in align_result:
                    assert(len(al) == topk), f"{i}, {align_result}, {al}"

                temp_dict["hyps"] = align_result

                if ('score' in d.keys()):
                    temp_dict["score"] = d["score"][:topk]
                if ('err' in d.keys()):
                    temp_dict["err"] = d["err"][:topk]
                temp_dict["top_hyp"] = d['hyps'][0] 
                temp_dict["ref"] = d["ref"]
                
                result_dict.append(temp_dict)
            
            if (use_concat):
                name = 'align_concat'
            else:
                name = 'align'

            if (task in ['dev', 'dev_ios']):
                save_name = [task, 'valid']
            else:
                save_name =[task]
            
            for to_save in save_name:
                save_path = Path(f"../data/{dataset}/{s}/{to_save}")
                save_path.mkdir(exist_ok=True, parents=True)

                with open(
                    f"{save_path}/{topk}_{name}_token.json", "w"
                ) as f:
                    json.dump(result_dict, f, ensure_ascii=False, indent=2)

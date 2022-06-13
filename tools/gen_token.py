import torch
import json
import logging
from transformers import BertTokenizer, BartTokenizer
import os

FORMAT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.DEBUG, filename="./log/gen_token.log", filemode="w", format=FORMAT
)


dataset = ["train", "dev", "test"]  # train
model_name = "bert"
nbest = 50


if model_name == "bart":
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
elif model_name == "bert":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

for d in dataset:
    print(d)
    if not os.path.exists(f"./data/aishell_{d}/{nbest}_best/{model_name}_token"):
        os.mkdir(f"./data/aishell_{d}/{nbest}_best/{model_name}_token")
    json_file = f"./data/aishell_{d}/{nbest}_best/dataset.json"
    w_json = f"./data/aishell_{d}/{nbest}_best/{model_name}_token/token.json"
    with open(json_file, "r") as f, open(w_json, "w") as fw:
        j = json.load(f)
        for i, element in enumerate(j):
            ids = []
            # seg = []
            text = []
            for seq in element["token"]:
                tokens = tokenizer.tokenize(seq)
                text.append(seq[5:-5])
                ids.append(tokenizer.convert_tokens_to_ids(tokens))
                # seg.append([0] * len(ids[-1]))
                # logging.warning(seg)
            # element["name"] = f"{d}_{i}"
            element["token"] = ids
            element["text"] = text
            # element["segment"] = seg

            ref_token = tokenizer.tokenize(element["ref"])
            element["ref_token"] = tokenizer.convert_tokens_to_ids(ref_token)
            element["ref_text"] = element["ref"][5:-5]
            # element["ref_seg"] = [0] * len(element["ref_token"])
        json.dump(j, fw, ensure_ascii=False, indent=4)

import torch
import json
import logging
from transformers import BertTokenizer, BartTokenizer
import os

FORMAT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.DEBUG, filename="./log/gen_token.log", filemode="w", format=FORMAT
)

setting = ["withLM", "noLM"]
dataset = ["train", "dev", "test"]  # train
model_name = "bart"
nbest = 50


if model_name == "bart":
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
elif model_name == "bert":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

logging.warning(f"start")
for s in setting:
    for d in dataset:
        print(f'{d}:{s}')
        if not os.path.exists(f"./data/aishell/{d}/token"):
            os.mkdir(f"./data/aishell/{d}/token")
        json_file = f"./data/aishell/{d}/data/data_{s}.json"
        w_json = f"./data/aishell/{d}/token/{model_name}_token_{s}.json"
        with open(json_file, "r") as f, open(w_json, "w") as fw:
            j = json.load(f)
            for i, element in enumerate(j):
                ids = []
                text = []
                for seq in element["token"]:
                    token = tokenizer.tokenize("[CLS]" + seq + "[SEP]")
                    text.append(seq)
                    ids.append(tokenizer.convert_tokens_to_ids(token))
                element["token"] = ids
                element["text"] = text

                ref_token = tokenizer.tokenize("[CLS]" + element["ref"] + "[SEP]")
                element["ref_token"] = tokenizer.convert_tokens_to_ids(ref_token)
                element["ref_text"] = element["ref"]

            logging.warning(f"write file")
            json.dump(j, fw, ensure_ascii=False, indent=4)

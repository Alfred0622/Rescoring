import torch
import json
import logging
from transformers import BertTokenizer, BartTokenizer
import os
from tqdm import tqdm

FORMAT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.DEBUG, filename="./log/gen_token.log", filemode="w", format=FORMAT
)

setting = ["withLM", "noLM"]
dataset = [ "dev", "test","train"]  # train
model_name = "bert"
task = "Correction"
nbest = 50
data_name = 'aishell'
addidtional_name = "/3align_concat/"


if (data_name in ['aishell', 'aishell2']):
    if model_name == "bart":
        tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    elif model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
elif (data_name in ['tedlium2', 'librispeech']):
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

logging.warning(f"start")
for s in setting:
    for d in dataset:
        print(f'{d}:{s}')
        if not os.path.exists(f"./src/{task}/data/{data_name}/{s}/{d}/token"):
            os.makedirs(f"./src/{task}/data/{data_name}/{s}/{d}/token")
        json_file = f"./data/{data_name}/data/{s}/{d}/data.json"
        w_json = f"./src/{task}/data/{data_name}/{s}/{d}/token/token.json"
        with open(json_file, "r") as f, open(w_json, "w") as fw:
            j = json.load(f)
            for i, element in enumerate(tqdm(j)):
                ids = []
                text = []
                for seq in element["hyp"]:
                    token = tokenizer.tokenize("[CLS] " + seq + " [SEP]")
                    text.append(seq)
                    ids.append(tokenizer.convert_tokens_to_ids(token))
                element["token"] = ids
                element["text"] = text

                ref_token = tokenizer.tokenize("[CLS] " + element["ref"] + " [SEP]")
                element["ref_token"] = tokenizer.convert_tokens_to_ids(ref_token)
                element["ref_text"] = element["ref"]

            logging.warning(f"write file")
            json.dump(j, fw, ensure_ascii=False, indent=4)

import torch
import sys
import json
from models.nBestAligner.nBestAlign import align_with_ref
from transformers import BertTokenizer
from tqdm import tqdm


task = ["dev", "test", "train"]
espnet_path = f"/mnt/nas3/Alfred/espnet/egs/aishell/asr1/dump/"

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
placeholder = "*"

for t in task:
    print(t)
    audio_path = f"{espnet_path}/{t}/deltafalse/data.json"
    nbest_path = f"./data/aishell_{t}/50_best/bert_token/token.json"

    with open(audio_path) as audio, open(nbest_path) as text, open(
        f"./data/aishell_{t}/50_best/bert_token/audio_data.json", "w"
    ) as comb:

        audio_json = json.load(audio)["utts"]
        text_json = json.load(text)

        combine_dict = list()

        for d in tqdm(text_json):
            single_dict = dict()
            name = d["name"]
            single_dict["name"] = name
            single_dict["feat"] = audio_json[name]["input"][0]["feat"]
            single_dict["feat_shape"] = audio_json[name]["input"][0]["shape"]

            single_dict["nbest"] = d["text"]
            single_dict["ref"] = d["ref"][5:-5]

            token = d["token"]
            ref_token = d["ref_token"]

            align_pair = align_with_ref(
                token,
                ref_token,
                placeholder=tokenizer.convert_tokens_to_ids(placeholder),
            )

            align_tokens = []
            align_ref_tokens = []

            for pair in align_pair:
                align_token = []
                align_ref_token = []
                for p in pair:
                    align_token.append(p[1])
                    align_ref_token.append(p[0])
                align_tokens.append(align_token)
                align_ref_tokens.append(align_ref_token)

            single_dict["nbest_token"] = align_tokens
            single_dict["ref_token"] = align_ref_tokens

            combine_dict.append(single_dict)

        json.dump(combine_dict, comb, ensure_ascii=False, indent=4)

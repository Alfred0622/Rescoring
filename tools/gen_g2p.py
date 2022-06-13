from g2pc import G2pC
from transformers import BertTokenizer
import torch
import torch.nn as nn
import json
from tqdm import tqdm

g2p = G2pC()
tokenizer = BertTokenizer.from_pretrained(f"bert-base-chinese")

recog_set = ["train", "dev", "test"]

for task in recog_set:
    json_file = f"./data/aishell_{task}/10_best/dataset.json"
    w_json = f"./data/aishell_{task}/10_best/bert_token/token_pho.json"
    print(f"{task}")

    with open(json_file) as f, open(w_json, "w") as w:
        data = json.load(f)
        for i, element in enumerate(tqdm(data)):
            ids = []
            # seg = []
            text = []
            pho = []
            for seq in element["token"]:
                tokens = tokenizer.tokenize(seq)
                text.append(seq[5:-5])
                phoneme = g2p(seq[5:-5])
                p_seq = ""
                for p in phoneme:
                    p_seq += f"{p[2]} "
                pho.append(p_seq)
                ids.append(tokenizer.convert_tokens_to_ids(tokens))
                # seg.append([0] * len(ids[-1]))
                # logging.warning(seg)
            element["name"] = f"{task}_{i}"
            element["token"] = ids
            element["text"] = text
            element["phoneme"] = pho
            # element["segment"] = seg

            ref_token = tokenizer.tokenize(element["ref"])
            element["ref_token"] = tokenizer.convert_tokens_to_ids(ref_token)
            element["ref_text"] = element["ref"][5:-5]
            # element["ref_seg"] = [0] * len(element["ref_token"])
        json.dump(data, w, ensure_ascii=False, indent=4)

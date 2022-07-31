import json
import yaml
import random
import torch
import glob
import logging
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.ComparisonRescoring.BertForComparison import BertForComparison
from utils.Datasets import(
    nBestDataset,
    rescoreDataset,
    concatDataset
)
from utils.CollateFunc import(
    bertCompareBatch,
    bertCompareRecogBatch,
)
from utils.LoadConfig import load_config

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/compare/train.log",
    filemode="w",
    format=FORMAT,
)

"""Basic setting"""
# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

config = f"./config/comparison.yaml"
args, train_args, recog_args = load_config(config)

setting = 'withLM' if args['withLM'] else 'noLM'

print(f"stage:{args['stage']}, stop_stage:{args['stop_stage']}")

# Prepare Data
print('Data Prepare')

# training file is too large, only load dev and test data here
with open(train_args['dev_json'], 'r') as dev, \
     open(train_args['test_json'], 'r') as test:
     dev_json = json.load(dev)
     test_json = json.load(test)

dev_dataset = rescoreDataset(dev_json, nbest = 50)
test_dataset = rescoreDataset(test_json, nbest = 50)


dev_loader = DataLoader(
    dev_dataset,
    batch_size = recog_args["batch"],
    collate_fn=bertCompareRecogBatch,
    pin_memory=True,
    num_workers=4,
)

test_loader = DataLoader(
    test_dataset,
    batch_size = recog_args["batch"],
    collate_fn=bertCompareRecogBatch,
    pin_memory=True,
    num_workers=4,
)


if (args['stage'] <= 0) and (args['stop_stage']>= 0):
    model = BertForComparison(
        lr = 1e-5
    ).to(device)

    print(f'training')
    min_loss = 1e8
    loss_seq = []
    train_path = f"{train_args['train_json']}/{setting}"
    for e in range(train_args["epoch"]):
        with open(f"{train_path}/token_concat.json", 'r') as f:
            train_json = json.load(f)
        train_dataset = concatDataset(
            train_json[:train_args["train_batch"]]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size = train_args["train_batch"],
            collate_fn=bertCompareBatch,
            pin_memory=True,
            num_workers=4,
        )
        for n, data in enumerate(tqdm(train_loader)):
            logging_loss = 0.0
            model.train()
            token, seg, masks, labels = data
            token = token.to(device)
            seg = seg.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            loss = model(token, seg, masks, labels)
            loss = loss / train_args["accumgrad"]
            loss.backward()
            
            logging_loss += loss.clone().detach().cpu()

            if ((n + 1) % train_args["accumgrad"] == 0) or ((n + 1) == len(train_loader)):
                model.optimizer.step()
                model.optimizer.zero_grad()

            if (n + 1) % train_args["print_loss"] == 0:
                logging.warning(
                    f"Training epoch :{e + 1} step:{n + 1}, loss:{logging_loss}"
                )
                loss_seq.append(logging_loss / train_args["print_loss"])
                logging_loss = 0.0
        
        train_checkpoint = dict()
        train_checkpoint["state_dict"] = model.model.state_dict()
        train_checkpoint["optimizer"] = model.optimizer.state_dict()
        if (not os.path.exists(f'./checkpoint/Comparision/BERT/{setting}')):
            os.makedirs(f'./checkpoint/Comparision/BERT/{setting}')
        
        torch.save(
            train_checkpoint,
            f"./checkpoint/Comparision/BERT/{setting}/checkpoint_train_{e + 1}.pt",
        )

        # eval
        model.eval()
        valid_loss = 0.0
        for n, data in enumerate(tqdm(dev_loader)):
            tokens, segs, masks, _, _, _, _, _, labels = data
            logging.warning(f'token.shape:{tokens.shape}')
            tokens = tokens.to(device)
            segs = segs.to(device)
            masks = masks.to(device)
            loss = model(tokens, segs, masks, labels)

            valid_loss += loss.clone().detach().cpu()
        
        if (valid_loss < min_loss):
            torch.save(
                train_checkpoint,
                f"./checkpoint/Comparision/BERT/{setting}/checkpoint_train_best.pt",
            )

            min_loss = valid_loss

if (args['stage'] <= 1) and (args['stop_stage'] >= 1):
    recog_set = ['dev', 'test']
    print(f'scoring')

    for task in recog_set:
        if (task == 'dev'):
            score_loader = dev_loader
        elif (task == 'test'):
            score_loader = test_loader
    
        recog_dict = []
        for n, data in enumerate(tqdm(score_loader)):
            tokens, segs, masks, first_score, errs, pairs, scores, texts = data
            output = model.recognize(tokens, segs, masks)
        
            for i, pair in enumerate(pairs):
                scores[pair[0]] += output[i][0]
                scores[pair[1]] += (1 - output[i][0])
            
            recog_dict.append(
                {
                    "token": tokens.tolist(),
                    "ref": texts,
                    "cer": errs,
                    "first_score": first_score.tolist(),
                    "rescore": scores.tolist(),
                }
            )

        if (not os.path.exists(f'data/aishell/{task}/BertCompare/{setting}')):
                os.makedirs(f'data/aishell/{task}/BertCompare/{setting}')
    
        print(f"writing file: ./data/aishell/{task}/BertCompare/{setting}/{nbest}best_recog_data.json")
        with open(
            f"./data/aishell/{task}/BertCompare/{setting}/{nbest}best_recog_data.json",
                "w"
        ) as f:
            json.dump(recog_dict, f, ensure_ascii=False, indent=4)
    
if (args['stage'] <= 2) and (args['stop_stage'] >= 2):
    print(f'rescoring')
    best_weight = 0.0
    with open(f"./data/aishell/dev/BertCompare/{setting}/{nbest}best_recog_data.json") as f:
        recog_file = json.load(f)

        # find best weight
        best_weight = 0.0
        min_err = 100
        for w in range(101):
            weight = w * 0.01
            for data in recog_file:
                first_score = torch.tensor(data['first_score'])
                rescore = torch.tensor(data['rescore'])
                cer = data['cer']
                cer = cer.view(-1, 4)

                weighted_score = first_score + weight * rescore

                max_index = torch.argmax(weighted_score).item()

                correction += cer[max_index][0]
                substitution += cer[max_index][1]
                deletion += cer[max_index][2]
                insertion += cer[max_index][3]

                err_for_weight = (substitution + deletion + insertion) / (
                        correction + deletion + substitution
                    )
                if (err_for_weight <= min_err):
                    print(f'better_weight:{min_weight}, smaller_err:{min_err}')
                    min_weight = weight
                    min_err = err_for_weight
                print(f'min_weight:{min_weight}, min_err:{min_err}')
        
    for task in recog_set:
        with open(f"./data/aishell/{task}/BertCompare/{setting}/{nbest}best_recog_data.json") as f:       
            recog_file = json.load(f)

            recog_dict = dict()
        recog_dict["utts"] = dict()
        for n, data in enumerate(recog_file):
            token = data["token"][:nbest]
            ref = data["ref"]

            score = torch.tensor(data["first_score"][:nbest])
            rescore = torch.tensor(data["rescore"][:nbest])

            weight_sum = score + best_weight * rescore

            max_index = torch.argmax(weight_sum).item()

            best_hyp = token[max_index]

            sep = best_hyp.index(102)
            best_hyp = tokenizer.convert_ids_to_tokens(t for t in best_hyp[1:sep])
            ref = list(ref[0])
            # remove [CLS] and [SEP]
            token_list = [str(t) for t in best_hyp]
            ref_list = [str(t) for t in ref]
            recog_dict["utts"][f"{task}_{n + 1}"] = dict()
            recog_dict["utts"][f"{task}_{n + 1}"]["output"] = {
                "rec_text": "".join(token_list),
                "rec_token": " ".join(token_list),
                "first_score": score.tolist(),
                "second_score": rescore.tolist(),
                "rescore": weight_sum.tolist(),
                "text": "".join(ref_list),
                "text_token": " ".join(ref_list),
            }

        with open(
            f"data/aishell/{task}/BertCompare/{setting}/{nbest}best_rescore_data.json",
            "w",
        ) as f:
            json.dump(recog_dict, f, ensure_ascii=False, indent=4)





            



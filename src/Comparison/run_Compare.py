from asyncio import selector_events
import json
import yaml
import random
import torch
import glob
import logging
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model.BertForComparison import BertForComparison
from utils.Datasets import(
    concatDataset,
    compareRecogDataset
)
from utils.CollateFunc import(
    bertCompareBatch,
    bertCompareRecogBatch,
)
from utils.LoadConfig import load_config

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

if (not os.path.exists("./log")):
    os.makedirs("./log")
FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"

logging.basicConfig(
    level=logging.INFO,
    filename="./log/train.log",
    filemode="w",
    format=FORMAT,
)

"""Basic setting"""
# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

config = "./config/comparison.yaml"
args, train_args, recog_args = load_config(config)

setting = 'withLM' if args['withLM'] else 'noLM'

print(f"stage:{args['stage']}, stop_stage:{args['stop_stage']}")

# Prepare Data
print('Data Prepare')
print(f'setting:{setting}')
recog_set = ['dev', 'test']   


if (args['stage'] <= 0) and (args['stop_stage']>= 0):
    model = BertForComparison(
        lr = 1e-5
    ).to(device)

    print(f'training')
    min_loss = 1e8
    loss_seq = []
    train_path = f"{train_args['train_json']}/{setting}"
    valid_path = f"{train_args['valid_json']}/{setting}"

    with open(f"{train_path}/token_concat.json", 'r') as f ,\
         open(f"{valid_path}/token_concat.json", 'r') as v:
        train_json = json.load(f)
        valid_json = json.load(v)
        print(f"# of train data:{len(train_json)}")
        print(f"# of valid data:{len(valid_json)}")
        train_dataset = concatDataset(
            train_json
        )
        valid_dataset = concatDataset(
            valid_json
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size = train_args["train_batch"],
            collate_fn=bertCompareBatch,
            num_workers=4,
            shuffle = True
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size = train_args["train_batch"],
            collate_fn=bertCompareBatch,
            num_workers=4,
        )
    for e in range(train_args["epoch"]):
        
        for n, data in enumerate(tqdm(train_loader)):
            logging_loss = 0.0
            model.train()
            token, seg, masks, labels = data

            token = token.to(device).to(torch.int64)
            # logging.warning(f'token:{token}')
            seg = seg.to(device).to(torch.int64)
            masks = masks.to(device).to(torch.int64)
            # logging.warning(f'seg:{seg}')
            # logging.warning(f'masks:{masks}')
            labels = labels.to(device)
            # logging.warning(f'labels:{labels}')

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
                loss_seq.append(logging_loss)
                logging_loss = 0.0
        
        train_checkpoint = dict()
        train_checkpoint["state_dict"] = model.model.state_dict()
        train_checkpoint["optimizer"] = model.optimizer.state_dict()
        if (not os.path.exists(f'./checkpoint/{setting}')):
            os.makedirs(f'./checkpoint/{setting}')
        
        torch.save(
            train_checkpoint,
            f"./checkpoint/{setting}/checkpoint_train_{e + 1}.pt",
        )

        # eval
        model.eval()
        valid_loss = 0.0   
        with torch.no_grad():
            for n, data in enumerate(tqdm(valid_loader)):
                tokens, segs, masks, labels = data
                tokens = tokens.to(device)
                segs = segs.to(device).to(torch.int64)
                masks = masks.to(device).to(torch.int64)
                labels = labels.to(device)
                loss = model(tokens, segs, masks, labels)

                valid_loss += loss.clone().detach().cpu()
        
        if (valid_loss < min_loss):
            torch.save(
                train_checkpoint,
                f"./checkpoint/{setting}/checkpoint_train_best.pt",
            )

            min_loss = valid_loss

if (args['stage'] <= 1) and (args['stop_stage'] >= 1):
    print('prepare recog data')
    with open(f"{train_args['dev_json']}/{setting}/token.json", 'r') as dev, \
         open(f"{train_args['test_json']}/{setting}/token.json", 'r') as test:
        dev_json = json.load(dev)
        test_json = json.load(test)

    dev_dataset = compareRecogDataset(dev_json)
    test_dataset = compareRecogDataset(test_json)


    dev_loader = DataLoader(
        dev_dataset,
        batch_size = recog_args["batch"],
        collate_fn=bertCompareRecogBatch,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = recog_args["batch"],
        collate_fn=bertCompareRecogBatch,
        num_workers=4,
    )

    recog_set = ['dev', 'test']
    print(f'scoring')
    checkpoint = torch.load(
        f'./checkpoint/{setting}/checkpoint_train_best.pt'
    )

    model = BertForComparison(
        lr = 1e-5
    ).to(device)
    model.model.load_state_dict(checkpoint['state_dict'])

    for task in recog_set:
        if (task == 'dev'):
            score_loader = dev_loader
        elif (task == 'test'):
            score_loader = test_loader
    
        recog_dict = []
        for n, data in enumerate(tqdm(score_loader)):
            name, tokens, segs, masks, pairs, texts, first_score, errs, ref, score= data
            tokens = tokens.to(device).to(torch.int64)
            segs = segs.to(device).to(torch.int64)
            masks = masks.to(device).to(torch.int64)

            output = model.recognize(tokens, segs, masks).clone().detach().cpu()
            for i, pair in enumerate(pairs):
                score[pair[0]] += output[i][0]
                score[pair[1]] += (1 - output[i][0])
            
            recog_dict.append(
                {
                    "name": name,
                    "text": texts,
                    "ref": ref,
                    "cer": errs,
                    "first_score": first_score,
                    "rescore": score.tolist(),
                }
            )

        if (not os.path.exists(f'./data/aishell/{task}/{setting}')):
                os.makedirs(f'./data/aishell/{task}/{setting}')
    
        print(f"writing file: ./data/aishell/{task}/{setting}/recog_data.json")
        with open(
            f"./data/aishell/{task}/{setting}/recog_data.json",
                "w"
        ) as f:
            json.dump(recog_dict, f, ensure_ascii=False, indent=4)
    
if (args['stage'] <= 2) and (args['stop_stage'] >= 2):
    print(f'rescoring')
    am_weight = 0.0
    lm_weight = 0.0
    with open(f"./data/aishell/dev/{setting}/recog_data.json") as f:
        recog_file = json.load(f)

        # find best weight
        best_lm = 0.0
        min_err = 100
        for l in tqdm(range(1001)):
            
            correction = 0
            substitution = 0
            deletion = 0
            insertion = 0

            l_weight = l * 0.01
            for data in recog_file:
                first_score = torch.tensor(data['first_score'][:args['nbest']])
                rescore = torch.tensor(data['rescore'])
                # print(first_score.shape)
                # print(rescore.shape)
                cer = torch.tensor(data['cer']).view(-1, 4)

                weighted_score = 1 * first_score + l_weight * rescore

                max_index = torch.argmax(weighted_score).item()

                correction += cer[max_index][0]
                substitution += cer[max_index][1]
                deletion += cer[max_index][2]
                insertion += cer[max_index][3]

            err_for_weight = (substitution + deletion + insertion) / (
                    correction + deletion + substitution
                )
            if (err_for_weight < min_err):
                    # print(f'better_weight:{a_weight}, {l_weight} ;  smaller_err:{min_err}')
                best_lm = l_weight
                min_err = err_for_weight
        print(f'min_weight:{l_weight}, min_err:{min_err}')
    
    print(f'using best_weight:{best_lm}')
    for task in recog_set:
        correction = 0
        substitution = 0
        deletion = 0
        insertion = 0

        with open(f"./data/aishell/{task}/{setting}/recog_data.json") as f:       
            recog_file = json.load(f)

            recog_dict = dict()
            recog_dict["utts"] = dict()
            for n, data in enumerate(recog_file):
                texts = data["text"]
                ref = data["ref"]

                score = torch.tensor(data["first_score"][:args['nbest']])
                rescore = torch.tensor(data["rescore"])
                cer = torch.tensor(data['cer']).view(-1, 4)

                weight_sum = 1 * score + best_lm * rescore

                max_index = torch.argmax(weight_sum).item()

                best_hyp = texts[max_index]

                correction += cer[max_index][0]
                substitution += cer[max_index][1]
                deletion += cer[max_index][2]
                insertion += cer[max_index][3]

                token_list = [str(t) for t in best_hyp]
                ref_list = [str(t) for t in ref]
                recog_dict["utts"][f"{task}_{n + 1}"] = dict()
                recog_dict["utts"][f"{task}_{n + 1}"]["output"] = {
                    "hyp": " ".join(token_list),
                    "first_score": score.tolist(),
                    "second_score": rescore.tolist(),
                    "rescore": weight_sum.tolist(),
                    "ref": " ".join(ref_list),
                }
            err = (substitution + deletion + insertion) / (
                    correction + deletion + substitution
                )
            print(f'{setting} / {task} : {err}')

        with open(
            f"./data/aishell/{task}/{setting}/BertSem_recog_data.json",
            "w",
        ) as f:
            json.dump(recog_dict, f, ensure_ascii=False, indent=4)





            



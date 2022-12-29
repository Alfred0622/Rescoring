import os
from tqdm import tqdm
import random
import json
import yaml
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from model.RescoreBert import RescoreBert, MLMBert
from transformers import BertTokenizer
from utils.Datasets import (
    adaptionDataset,
    pllDataset,
    rescoreDataset
) 
from utils.CollateFunc import(
    adaptionBatch,
    pllScoringBatch,
    rescoreBertBatch,
    RescoreBertRecog
)
from utils.LoadConfig import load_config

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

"""Basic setting"""
# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device:{device}')

config = f"./config/RescoreBert.yaml"

args, adapt_args, train_args, recog_args = load_config(config)

print(f"stage:{args['stage']}, stop_stage:{args['stop_stage']}")

if adapt_args["mode"] == "sequence":
    adapt_batch = adapt_args["mlm_batch"]
else:
    adapt_batch = adapt_args["train_batch"]

# training
use_MWER = False
use_MWED = False
print(f"training mode:{train_args['mode']}")
print(f"conf nBest:{args['nbest']}")

if train_args['mode'] == "MWER":
    use_MWER = True
elif train_args['mode'] == "MWED":
    use_MWED = True

scoring_set = ["train", "dev", "test"]

setting = "withLM" if args['withLM'] else "noLM"
""""""

if (not os.path.exists(f"./log/{train_args['mode']}/{setting}")):
    os.makedirs(f"./log/{train_args['mode']}/{setting}")

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/{train_args['mode']}/{setting}/train.log",
    filemode="w",
    format=FORMAT,
)


print(f"Prepare data")
train_json_name = f"./data/{args['dataset']}best/{setting}/50best/MLM/train/rescore_data.json"
dev_json_name = f"./data/{args['dataset']}best/{setting}/50best/MLM/dev/rescore_data.json"
test_json_name = f"./data/{args['dataset']}best/{setting}/50best/MLM/test/rescore_data.json"



"""Training Dataloader"""
if train_args['mode'] == "SimCSE":
    pass

"""Init model"""
logging.warning(f"device:{device}")
device = torch.device(device)

pretrain_name = "bert-base-chinese"
if (args['dataset'] in ['aishell', 'aishell2']):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
elif (args['dataset'] in ['tedlium2', 'librispeech']):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    pretrain_name = "bert-base-uncased"
elif (args['dataset'] in ['csj']):
    pass  #japanese

print(f'pretrain_name:{pretrain_name}')

if args['stage'] <= 2:
    model = MLMBert(
        train_batch=adapt_batch,
        test_batch=recog_args["batch"],
        nBest=args["nbest"],
        device=device,
        mode=adapt_args["mode"],
        lr=float(adapt_args["lr"]),
        pretrain_name = pretrain_name
    )

if args['stage'] <= 1 and args['stop_stage'] >= 1:
    # Prepare data
    with open(train_json_name) as f:
        train_json = json.load(f)
    adaption_set = adaptionDataset(train_json)

    adaption_loader = DataLoader(
        dataset=adaption_set,
        batch_size=adapt_batch,
        collate_fn=adaptionBatch,
        pin_memory=True,
        num_workers=4,
    )

    adapt_loss = []
    adapt_checkpoint = dict()

    print(f"domain adaption : {adapt_args['mode']} mode")
    model.optimizer.zero_grad()
    if adapt_args['epoch'] > 0:
        for e in range(adapt_args['epoch']):
            model.train()
            logging_loss = 0.0
            for n, data in enumerate(tqdm(adaption_loader)):
                token, mask = data
                token = token.to(device)
                mask = mask.to(device)

                loss = model(token, mask)
                loss = loss / train_args["accumgrad"]
                loss.backward()
                logging_loss += loss.item()

                if ((n + 1) % train_args["accumgrad"] == 0) or ((n + 1) == len(adaption_loader)):
                    model.optimizer.step()
                    model.optimizer.zero_grad()

                if (n + 1) % train_args["print_loss"] == 0:
                    logging.warning(
                        f"Adaption epoch :{e + 1} step:{n + 1}, adaption loss:{logging_loss}"
                    )
                    adapt_loss.append(logging_loss / train_args["print_loss"])
                    logging_loss = 0.0

            adapt_checkpoint["state_dict"] = model.model.state_dict()
            adapt_checkpoint["optimizer"] = model.optimizer.state_dict()
            if (not os.path.exists(f"./checkpoint/{args['dataset']}/adaption")):
                os.makedirs(f"./checkpoint/{args['dataset']}/adaption")
            torch.save(
                adapt_checkpoint,
                f"./checkpoint/{args['dataset']}/adaption/checkpoint_adapt_{e + 1}.pt",
            )

        if not os.path.exists(f"./log/RescoreBert/{args['dataset']}"):
            os.makedirs(f"./log/RescoreBert/{args['dataset']}")
        torch.save(adapt_loss, f"./log/RescoreBert/{args['dataset']}/adaption_loss.pt")

if args['stage'] <= 2 and args['stop_stage'] >= 2:
    """PLL Scoring"""

    print(f"PLL scoring, loading data:")

    print(f"start scoring")
    
    if args['stage'] == 2:
        print(
            f"using checkpoint:./checkpoint/{args['dataset']}/adaption/checkpoint_adapt_{adapt_args['epoch']}.pt"
        )
        adapt_checkpoint = torch.load(
            f"./checkpoint/{args['dataset']}/adaption/checkpoint_adapt_{adapt_args['epoch']}.pt"
        )
        model.model.load_state_dict(adapt_checkpoint["state_dict"])

    model.eval()

    for t in scoring_set:
        if t == "train":
            with open(train_json_name) as f:
                pll_data = json.load(f)
                pll_dataset = rescoreDataset(pll_data, args["nbest"])
                pll_loader = DataLoader(
                             dataset=pll_dataset,
                             batch_size=recog_args["batch"],
                             collate_fn=pllScoringBatch,
                             num_workers=4,
                )
        elif t == "dev":
            with open(dev_json_name) as f:
                pll_data = json.load(f)
                pll_dataset = rescoreDataset(pll_data, args["nbest"])
                pll_loader = DataLoader(
                             dataset=pll_dataset,
                             batch_size=recog_args["batch"],
                             collate_fn=pllScoringBatch,
                             pin_memory=True,
                             num_workers=4,
                )

        elif t == "test":
            with open(test_json_name) as f:
                pll_data = json.load(f)
                pll_dataset = rescoreDataset(pll_data, args["nbest"])
                pll_loader = DataLoader(
                             dataset=pll_dataset,
                             batch_size=recog_args["batch"],
                             collate_fn=pllScoringBatch,
                             pin_memory=True,
                             num_workers=4,
                )
        
        data_name_dict = dict()
        for i, data in enumerate(pll_data):
            data_name_dict[data["name"]] = i

        with torch.no_grad():
            for n, data in enumerate(tqdm(pll_loader)):
                name, token, text, mask, _, _, _ = data
                logging.warning(f'name:{name}')

                token = token.to(device)
                mask = mask.to(device)

                pll_score = model.recognize(
                    token,
                    mask, 
                    free_memory = args['free_memory'],
                    sentence_per_process = args['sentence_per_process']
                )
                # train_json during training
                pll_data[data_name_dict[name]]['pll'] = pll_score.tolist()
                # for i, data in enumerate(pll_data):
                #     if data["name"] == name:
                #         # train_json during training
                #         pll_data[i]["pll"] = pll_score.tolist()

        # debug
        for i, data in enumerate(pll_data):
            assert "pll" in data.keys(), "PLL score not exist."
            assert len(data['pll']) == len(data['score']), "length of pll and score should be same"
        print(f"saving file at ./data/{args['dataset']}/{setting}/{t}/pll_data/token_pll.json")
        if (not os.path.exists(f"./data/{args['dataset']}/{setting}/{t}/pll_data/")):
            os.makedirs(f"./data/{args['dataset']}/{setting}/{t}/pll_data/")
        with open(
            f"./data/{args['dataset']}/{setting}/{t}/pll_data/token_pll.json", "w"
        ) as f:
            json.dump(pll_data, f, ensure_ascii=False, indent=4)



if args['stop_stage'] >= 3:
    model = RescoreBert(
        train_batch=train_args['train_batch'],
        test_batch=recog_args["batch"],
        nBest=args["nbest"],
        use_MWER=use_MWER,
        use_MWED=use_MWED,
        device=device,
        lr=float(train_args["lr"]),
        weight=0.59,
        pretrain_name = pretrain_name
    )


"""training"""
if args['stage'] <= 3 and args['stop_stage'] >= 3:
    train_json = None
    valid_json = None
    pll_train_file = f"./data/{args['dataset']}/{setting}/train/pll_data/token_pll.json"
    print(f'loading training file from: {pll_train_file}')

    with open(pll_train_file, "r") as f:
        train_json = json.load(f)

    train_set = pllDataset(train_json, nbest=args["nbest"])
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_args['train_batch'],
        collate_fn=rescoreBertBatch,
        pin_memory=True,
        num_workers=4,
    )

    pll_dev_file = f"./data/{args['dataset']}/{setting}/dev/pll_data/token_pll.json"
    print(f'loading dev file from:{pll_dev_file}')

    with open(pll_dev_file, "r") as f:
        valid_json = json.load(f)
    valid_set = pllDataset(valid_json, nbest=args["nbest"])
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=recog_args["batch"], collate_fn=rescoreBertBatch
    )
    print(f"training...")
    
    last_val = 1e8
    train_loss = []
    dev_loss = []
    dev_cers = []
    min_epoch = train_args['epoch']
    model.optimizer.zero_grad()
    for e in range(train_args['epoch']):
        train_checkpoint = dict()
        model.train()
        accum_loss = 0.0
        logging_loss = 0.0
        model.optimizer.zero_grad()
        for n, data in enumerate(tqdm(train_loader)):
            token, text, mask, score, cer, pll = data
            token = token.to(device)
            mask = mask.to(device)
            score = score.to(device)
            cer = cer.to(device)
            pll = pll.to(device)

            loss = model(token, text, mask, score, cer, pll)
            loss = loss / train_args["accumgrad"]
            # logging.warning(f"loss:{loss}")
            loss.backward()

            logging_loss += loss.clone().detach().cpu()

            if (n + 1) % train_args["accumgrad"] == 0 or (n + 1) == len(train_loader):
                model.optimizer.step()
                model.optimizer.zero_grad()

            if (n + 1) % train_args["print_loss"] == 0 or (n + 1) == len(train_loader):
                train_loss.append(logging_loss / train_args["print_loss"])
                logging.warning(
                    f"Training epoch:{e + 1} step:{n + 1}, training loss:{logging_loss / train_args['print_loss']}"
                )
                logging_loss = 0.0
        
        train_checkpoint['epoch'] = e + 1
        train_checkpoint["state_dict"] = model.model.state_dict()
        train_checkpoint["optimizer"] = model.optimizer.state_dict()
        if (not os.path.exists(f"./checkpoint/{train_args['mode']}/{setting}")):
            os.makedirs(f"./checkpoint/{train_args['mode']}/{setting}")
        
        torch.save(
            train_checkpoint,
            f"./checkpoint/{train_args['mode']}/{setting}/checkpoint_train_{e + 1}.pt",
        )

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_cer = 0.0
            c = 0
            s = 0
            d = 0
            i = 0
            for n, data in enumerate(tqdm(valid_loader)):
                token, text, mask, score, cer, pll = data
                token = token.to(device)
                mask = mask.to(device)
                score = score.to(device)
                cer = cer.to(device)
                pll = pll.to(device)

                loss, err = model(token, text, mask, score, cer, pll)
                val_loss += loss
                c += err[0]
                s += err[1]
                d += err[2]
                i += err[3]

            val_cer = (s + d + i) / (c + s + d)
            val_loss = val_loss / len(valid_loader)
            dev_loss.append(val_loss)
            dev_cers.append(val_cer)

        logging.warning(f"epoch :{e + 1}, validation_loss:{val_loss}")
        logging.warning(f"epoch :{e + 1}, validation_cer:{val_cer}")

        if val_loss < last_val:
            last_val = val_loss
            torch.save(
                train_checkpoint,
                f"./checkpoint/{train_args['mode']}/{setting}/checkpoint_train_best.pt",
            )

    logging_loss = {
        "training_loss": train_loss,
        "dev_loss": dev_loss,
        "dev_cer": dev_cers,
    }
    if not os.path.exists(f"./log/{train_args['mode']}/{setting}"):
        os.makedirs(f"./log/{train_args['mode']}/{setting}")
    torch.save(logging_loss, f"./log/{train_args['mode']}/{setting}/loss.pt")

# recognizing
if args['stage'] <= 4 and args['stop_stage'] >= 4:
    print(f"scoring")
    checkpoint = torch.load(
        f"./checkpoint/{train_args['mode']}/{setting}/checkpoint_train_best.pt"
    )
    print(f"best epoch : {checkpoint['epoch']}")
    model.model.load_state_dict(checkpoint["state_dict"])
    model.optimizer.load_state_dict(checkpoint["optimizer"])

    model.eval()
    recog_set = ["dev", "test"]
    recog_data = None
    with torch.no_grad():
        for task in recog_set:

            print(f"scoring: {task}")
            file = f"./data/{args['dataset']}/{setting}/{task}/pll_data/token_pll.json"
            with open(file, 'r') as f:
                recog_json = json.load(f)
            recog_dataset = rescoreDataset(recog_json, nbest = args['nbest'])
            recog_data = DataLoader(
                dataset=recog_dataset,
                batch_size=recog_args["batch"],
                collate_fn=RescoreBertRecog,
                pin_memory=True,
            )

            recog_dict = []
            for n, data in enumerate(tqdm(recog_data)):
                _, token, mask, score, texts, ref, cers = data
                token = token.to(device)
                mask = mask.to(device)
                score = torch.tensor(score).to(device)

                rescore, _, _, _ = model.recognize(token, texts, mask, score, weight=1)

                recog_dict.append(
                    {
                        "hyp": texts,
                        "ref": ref,
                        "cer": cers,
                        "first_score": score.tolist(),
                        "rescore": rescore.tolist(),
                    }
                )
            if (not os.path.exists(f"data/{args['dataset']}/{setting}/{task}/{train_args['mode']}")):
                os.makedirs(f"data/{args['dataset']}/{setting}/{task}/{train_args['mode']}")
            print(f"writing file:./data/{args['dataset']}/{setting}/{task}/{train_args['mode']}/recog_data.json")
            with open(
                f"./data/{args['dataset']}/{setting}/{task}/{train_args['mode']}/recog_data.json",
                "w",
            ) as f:
                json.dump(recog_dict, f, ensure_ascii=False, indent=4)

if args['stage'] <= 5 and args['stop_stage'] >= 5:
    # find best weight
    if recog_args["find_weight"]:
        print(f"Finding Best weight")
        print(f"loading recog from: ./data/{args['dataset']}/{setting}/dev/{train_args['mode']}/recog_data.json")
        with open(
            f"./data/{args['dataset']}/{setting}/dev/{train_args['mode']}/recog_data.json"
        ) as f:
            val_score = json.load(f)

        best_cer = 100
        best_weight = 0
        for w in tqdm(range(101)):
            correction = 0  # correction
            substitution = 0  # substitution
            deletion = 0  # deletion
            insertion = 0  # insertion

            weight = w * 0.01
            for data in val_score:
                first_score = torch.tensor(data["first_score"][:args["nbest"]])
                rescore = torch.tensor(data["rescore"][:args["nbest"]])
                cer = torch.tensor(data["cer"][:args["nbest"]])
                cer = cer.view(-1, 4)

                weighted_score = first_score + weight * rescore

                max_index = torch.argmax(weighted_score).item()

                correction += cer[max_index][0]
                substitution += cer[max_index][1]
                deletion += cer[max_index][2]
                insertion += cer[max_index][3]

            cer = (substitution + deletion + insertion) / (
                correction + deletion + substitution
            )
            logging.warning(f"weight:{weight}, cer:{cer}")
            if best_cer > cer:
                print(f"update weight:{weight}, cer:{cer}\r")
                best_cer = cer
                best_weight = weight
    else:
        best_weight = 0.59

if args['stage'] <= 6 and args['stop_stage']>= 6:
    print("output result")
    if args['stage'] == 6:
        best_weight = 0.59
    print(f"Best weight at: {best_weight}")
    recog_set = ["dev", "test"]
    for task in recog_set:
        print(f"recogizing: {task}")
        score_data = None
        with open(
            f"./data/{args['dataset']}/{setting}/{task}/{train_args['mode']}/recog_data.json"
        ) as f:
            score_data = json.load(f)

        recog_dict = dict()
        recog_dict["utts"] = dict()

        c = 0
        s = 0
        i = 0
        d = 0

        for n, data in enumerate(score_data):
            token = data["hyp"][:args["nbest"]]
            ref = data["ref"]
            cer = torch.tensor(data["cer"][:args["nbest"]])
            cer = cer.view(-1, 4)

            score = torch.tensor(data["first_score"][:args["nbest"]])
            rescore = torch.tensor(data["rescore"][:args["nbest"]])

            weight_sum = score + best_weight * rescore

            max_index = torch.argmax(weight_sum).item()

            best_hyp = token[max_index]

            c += cer[max_index][0]
            s += cer[max_index][1]
            d += cer[max_index][2]
            i += cer[max_index][3]

            ref_str = str()
            for t in ref:
                ref_str += t

            recog_dict["utts"][f"{task}_{n + 1}"] = dict()
            recog_dict["utts"][f"{task}_{n + 1}"]["output"] = {
                "hyp": "".join(best_hyp),
                "ref": " ".join(ref),
                "first_score": score.tolist(),
                "second_score": rescore.tolist(),
                "rescore": weight_sum.tolist(),
                # "text": "".join(ref_list),
                # "text_token": " ".join(ref_list),
            }
        cer = (i + d + s) / (c + d + s)
        print(f'cer:{cer}')

        with open(
            f"./data/{args['dataset']}/{setting}/{task}/{train_args['mode']}/{args['nbest']}best_rescore_data.json",
            "w",
        ) as f:
            json.dump(recog_dict, f, ensure_ascii=False, indent=4)

print("Finish")

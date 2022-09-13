import os
from tqdm import tqdm
import random
import json
import yaml
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from models.BertForRescoring.RescoreBert import RescoreBert, MLMBert
from transformers import BertTokenizer
from utils.Datasets import (
    adaptionDataset,
    pllDataset,
    rescoreDataset
) 
from utils.CollateFunc import(
    adaptionBatch,
    pllScoringBatch,
    rescoreBertBatch
)


random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

"""Basic setting"""
# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

config = f"./config/RescoreBert.yaml"
adapt_args = dict()
train_args = dict()
recog_args = dict()

with open(config, "r") as f:
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    stage = conf["stage"]
    nbest = conf["nbest"]
    stop_stage = conf["stop_stage"]
    withLM = conf["withLM"]
    dataset = conf["dataset"]
    # Adaption, training, recog args
    adapt_args = conf["adapt"]
    train_args = conf["train"]
    recog_args = conf["recog"]

print(f"stage:{stage}, stop_stage:{stop_stage}")
# adaption
adapt_epoch = adapt_args["epoch"]
adapt_lr = float(adapt_args["lr"])
adapt_mode = adapt_args["mode"]

if adapt_mode == "sequence":
    adapt_batch = adapt_args["mlm_batch"]
else:
    adapt_batch = adapt_args["train_batch"]

# training
epochs = train_args["epoch"]
train_batch = train_args["train_batch"]
accumgrad = train_args["accumgrad"]
print_loss = train_args["print_loss"]
train_lr = float(train_args["lr"])
training = train_args["mode"]
use_MWER = False
use_MWED = False
print(f"training mode:{training}")
print(f"conf nBest:{nbest}")
if training == "MWER":
    use_MWER = True
elif training == "MWED":
    use_MWED = True

# recognition
recog_batch = recog_args["batch"]
find_weight = recog_args["find_weight"]

""""""
adapt_checkpoint = {"state_dict": None, "optimizer": None, "last_val_loss": None}

train_checkpoint = {
    "training": None,
    "state_dict": None,
    "optimizer": None,
    "last_val_loss": None,
}
train_checkpoint["training"] = training
setting = "withLM" if withLM else "noLM"

if (not os.path.exists(f'./log/{training}/{setting}')):
    os.makedirs(f'./log/{training}/{setting}')

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/{training}/{setting}/train.log",
    filemode="w",
    format=FORMAT,
)

print(f"Prepare data")
train_json = f"./data/{dataset}/{setting}/train/token/token.json"
dev_json = f'./data/{dataset}/{setting}/dev/token/token.json'
test_json = f'./data/{dataset}/{setting}/test/token/token.json'

with open(train_json) as f, open(dev_json) as d, open(test_json) as t:
    train_json = json.load(f)
    dev_json = json.load(d)
    test_json = json.load(t)

adaption_set = adaptionDataset(train_json)
train_set = rescoreDataset(train_json, nbest)
dev_set = rescoreDataset(dev_json, nbest)
test_set = rescoreDataset(test_json, nbest)

"""Training Dataloader"""
adaption_loader = DataLoader(
    dataset=adaption_set,
    batch_size=adapt_batch,
    collate_fn=adaptionBatch,
    pin_memory=True,
    num_workers=4,
)

scoring_loader = DataLoader(
    dataset=train_set,
    batch_size=recog_batch,
    collate_fn=pllScoringBatch,
    pin_memory=True,
    num_workers=4,
)

dev_loader = DataLoader(
    dataset=dev_set,
    batch_size=recog_batch,
    collate_fn=pllScoringBatch,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=recog_batch,
    collate_fn=pllScoringBatch,
    pin_memory=True,
)

if training == "SimCSE":
    pass

"""Init model"""
logging.warning(f"device:{device}")
device = torch.device(device)

pretrain_name = "bert-base-chinese"
if (dataset in ['aishell', 'aishell2']):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
elif (dataset in ['tedlium2', 'librispeech']):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    pretrain_name = "bert-base-uncased"
elif (dataset in ['csj']):
    pass  #japanese

scoring_set = [ "dev", "test"]

if stage <= 2:
    model = MLMBert(
        train_batch=adapt_batch,
        test_batch=recog_batch,
        nBest=nbest,
        device=device,
        mode=adapt_mode,
        lr=adapt_lr,
        pretrain_name = pretrain_name
    )

if stage <= 1 and stop_stage >= 1:
    adapt_loss = []

    print(f"domain adaption : {adapt_mode} mode")
    model.optimizer.zero_grad()
    if adapt_epoch > 0:
        for e in range(adapt_epoch):
            model.train()
            logging_loss = 0.0
            for n, data in enumerate(tqdm(adaption_loader)):
                token, mask = data
                token = token.to(device)
                mask = mask.to(device)

                loss = model(token, mask)
                loss = loss / accumgrad
                loss.backward()
                logging_loss += loss.clone().detach().cpu()

                if ((n + 1) % accumgrad == 0) or ((n + 1) == len(adaption_loader)):
                    model.optimizer.step()
                    model.optimizer.zero_grad()

                if (n + 1) % print_loss == 0:
                    logging.warning(
                        f"Adaption epoch :{e + 1} step:{n + 1}, adaption loss:{logging_loss}"
                    )
                    adapt_loss.append(logging_loss / print_loss)
                    logging_loss = 0.0
            adapt_checkpoint["state_dict"] = model.model.state_dict()
            adapt_checkpoint["optimizer"] = model.optimizer.state_dict()
            torch.save(
                adapt_checkpoint,
                f"./checkpoint/RescoreBert/adaption/checkpoint_adapt_{e + 1}.pt",
            )

        if not os.path.exists("./log/RescoreBert"):
            os.makedirs("./log/RescoreBert")
        torch.save(adapt_loss, "./log/RescoreBert/adaption_loss.pt")

if stage <= 2 and stop_stage >= 2:
    """PLL Scoring"""
    print(f"PLL scoring:")
    if stage == 2:
        print(
            f'using checkpoint:"./checkpoint/RescoreBert/adaption/checkpoint_adapt_{adapt_epoch}.pt"'
        )
        adapt_checkpoint = torch.load(
            f"./checkpoint/RescoreBert/adaption/checkpoint_adapt_{adapt_epoch}.pt"
        )
        model.model.load_state_dict(adapt_checkpoint["state_dict"])

    model.eval()
    pll_data = train_json
    pll_loader = scoring_loader
    for t in scoring_set:
        if t == "train":
            pll_data = train_json
            pll_loader = scoring_loader

        elif t == "dev":
            pll_data = dev_json
            pll_loader = dev_loader

        elif t == "test":
            pll_data = test_json
            pll_loader = test_loader

        with torch.no_grad():
            for n, data in enumerate(tqdm(pll_loader)):
                name, token, text, mask, _, _, _ = data

                token = token.to(device)
                mask = mask.to(device)

                pll_score = model.recognize(token, mask, free_memory = True)
                # train_json during training
                for i, data in enumerate(pll_data):
                    if data["name"] == name:
                        # train_json during training
                        pll_data[i]["pll"] = pll_score.tolist()

            # debug
        for i, data in enumerate(pll_data):
            assert "pll" in data.keys(), "PLL score not exist."
            assert len(data['pll']) == len(data['score']), "length of pll and score should be same"
        print(f'saving file at ./data/aishell/{t}/pll_data/token_pll_{setting}.json')
        with open(
            f"./data/aishell/{t}/pll_data/token_pll_{setting}.json", "w"
        ) as f:
            json.dump(pll_data, f, ensure_ascii=False, indent=4)



if stop_stage >= 3:
    model = RescoreBert(
        train_batch=train_batch,
        test_batch=recog_batch,
        nBest=nbest,
        use_MWER=use_MWER,
        use_MWED=use_MWED,
        device=device,
        lr=train_lr,
        weight=0.59,
        pretrain_name = pretrain_name
    )
    
min_epoch = epochs
"""training"""
if stage <= 3 and stop_stage >= 3:
    train_json = None
    valid_json = None
    pll_train_file = f'./data/aishell/train/pll_data/token_pll_{setting}.json'
    print(f'loading training file from: {pll_train_file}')
    with open(pll_train_file, "r") as f:
        train_json = json.load(f)
    train_set = pllDataset(train_json, nbest=nbest)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_batch,
        collate_fn=rescoreBertBatch,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
    )
    pll_dev_file = f"./data/aishell/dev/pll_data/token_pll_{setting}.json"
    print(f'loading dev file from:{pll_dev_file}')
    with open(pll_dev_file, "r") as f:
        valid_json = json.load(f)
    valid_set = pllDataset(valid_json, nbest=nbest)
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=recog_batch, collate_fn=rescoreBertBatch
    )
    print(f"training...")
    model.optimizer.zero_grad()

    last_val = 1e8
    train_loss = []
    dev_loss = []
    dev_cers = []
    min_epoch = epochs
    for e in range(epochs):
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
            loss = loss / accumgrad
            # logging.warning(f"loss:{loss}")
            loss.backward()

            logging_loss += loss.clone().detach().cpu()

            if (n + 1) % accumgrad == 0 or (n + 1) == len(train_loader):
                model.optimizer.step()
                model.optimizer.zero_grad()

            if (n + 1) % print_loss == 0 or (n + 1) == len(train_loader):
                train_loss.append(logging_loss / print_loss)
                logging.warning(
                    f"Training epoch:{e + 1} step:{n + 1}, training loss:{logging_loss / print_loss}"
                )
                logging_loss = 0.0

        train_checkpoint["state_dict"] = model.model.state_dict()
        train_checkpoint["optimizer"] = model.optimizer.state_dict()
        if (not os.path.exists(f'./checkpoint/RescoreBert/{training}/{setting}')):
            os.makedirs(f'./checkpoint/RescoreBert/{training}/{setting}')
        
        torch.save(
            train_checkpoint,
            f"./checkpoint/RescoreBert/{training}/{setting}/checkpoint_train_{e + 1}.pt",
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
            val_loss = val_loss / len(dev_loader)
            dev_loss.append(val_loss)
            dev_cers.append(val_cer)

        logging.warning(f"epoch :{e + 1}, validation_loss:{val_loss}")
        logging.warning(f"epoch :{e + 1}, validation_cer:{val_cer}")

        if val_loss < last_val:
            last_val = val_loss
            min_epoch = e + 1
            stage = 4

    logging_loss = {
        "training_loss": train_loss,
        "dev_loss": dev_loss,
        "dev_cer": dev_cers,
    }
    if not os.path.exists(f"./log/RescoreBert/{training}/{setting}"):
        os.makedirs(f"./log/RescoreBert/{training}/{setting}")
    torch.save(logging_loss, f"./log/RescoreBert/{training}/{setting}/loss.pt")

# recognizing
if stage <= 4 and stop_stage >= 4:
    print(f"scoring")
    if stage == 4:
        print(
            f"using checkpoint: ./checkpoint/RescoreBert/{training}/{setting}/checkpoint_train_{min_epoch}.pt"
        )
        checkpoint = torch.load(
            f"./checkpoint/RescoreBert/{training}/{setting}/checkpoint_train_{min_epoch}.pt"
        )
        model.model.load_state_dict(checkpoint["state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer"])

    model.eval()
    recog_set = ["train", "dev", "test"]
    recog_data = None
    with torch.no_grad():
        for task in recog_set:
            print(f"scoring: {task}")
            if task == "train":
                recog_data = scoring_loader
            elif task == "dev":
                recog_data = dev_loader
            elif task == "test":
                recog_data = test_loader

            recog_dict = []
            for n, data in enumerate(tqdm(recog_data)):
                _, token, text, mask, score, ref, cer = data
                token = token.to(device)
                mask = mask.to(device)
                score = score.to(device)

                rescore, _, _, _ = model.recognize(token, text, mask, score, weight=1)

                recog_dict.append(
                    {
                        "token": token.tolist(),
                        "ref": ref,
                        "cer": cer,
                        "first_score": score.tolist(),
                        "rescore": rescore.tolist(),
                    }
                )
            if (not os.path.exists(f'data/aishell/{task}/{training}/{setting}')):
                os.makedirs(f'data/aishell/{task}/{training}/{setting}')
            print(f"writing file:./data/aishell/{task}/{training}/{setting}/{nbest}best_recog_data.json")
            with open(
                f"./data/aishell/{task}/{training}/{setting}/{nbest}best_recog_data.json",
                "w",
            ) as f:
                json.dump(recog_dict, f, ensure_ascii=False, indent=4)

if stage <= 5 and stop_stage >= 5:
    # find best weight
    if find_weight:
        print(f"Finding Best weight")
        print(f'loading recog from: ./data/aishell/dev/{training}/{setting}/{nbest}best_recog_data.json')
        with open(
            f"./data/aishell/dev/{training}/{setting}/{nbest}best_recog_data.json"
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
                first_score = torch.tensor(data["first_score"][:nbest])
                rescore = torch.tensor(data["rescore"][:nbest])
                cer = torch.tensor(data["cer"][:nbest])
                cer = cer.view(-1, 4)

                weighted_score = first_score + weight * rescore

                max_index = torch.argmax(weighted_score)

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

if stage <= 6 and stop_stage >= 6:
    print("output result")
    if stage == 6:
        best_weight = 0.59
    print(f"Best weight at: {best_weight}")
    recog_set = ["train", "dev", "test"]
    for task in recog_set:
        print(f"recogizing: {task}")
        score_data = None
        with open(
            f"data/aishell/{task}/{training}/{setting}/{nbest}best_recog_data.json"
        ) as f:
            score_data = json.load(f)

        recog_dict = dict()
        recog_dict["utts"] = dict()
        for n, data in enumerate(score_data):
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
            f"data/aishell/{task}/{training}/{setting}/{nbest}best_rescore_data.json",
            "w",
        ) as f:
            json.dump(recog_dict, f, ensure_ascii=False, indent=4)

print("Finish")

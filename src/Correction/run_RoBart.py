import os
from tqdm import tqdm
import random
import yaml
import logging
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.BartForCorrection.RoBart import RoBart
from utils.Datasets import correctDataset, correctRecogDataset
from utils.CollateFunc import correctBatch, correctRecogBatch
from transformers import BertTokenizer
from utils.LoadConfig import load_config

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

config = f"./config/RoBart.yaml"

args, train_args, recog_args = load_config(config)

setting = "withLM" if args["withLM"] else "noLM"

if (not os.path.exists(f"./log/{setting}/{args['nbest']}best")):
    os.makedirs(f"./log/{setting}/{args['nbest']}best")

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/{setting}/{args['nbest']}best/train.log",
    filemode="w",
    format=FORMAT,
)

train_json = None
dev_json = None
test_json = None

print(f"Prepare data")
print(f'Load json file')

train_path = f"./data/{args['dataset']}/{setting}/train/{args['nbest']}best/token.json"
valid_path = f"./data/{args['dataset']}/{setting}/valid/1best/token.json"
dev_path = f"./data/{args['dataset']}/{setting}/dev/1best/token.json"
test_path = f"./data/{args['dataset']}/{setting}/test/1best/token.json"

with open(train_path) as f,\
     open(valid_path) as v,\
     open(dev_path) as d,\
     open(test_path) as t:
    train_json = json.load(f)
    valid_json = json.load(v)
    dev_json = json.load(d)
    test_json = json.load(t)

print(f'Create Dataset & DataLoader')
train_set = correctDataset(train_json)
valid_set = correctDataset(valid_json)
dev_set = correctRecogDataset(dev_json)
test_set = correctRecogDataset(test_json)

train_loader = DataLoader(
    dataset=train_set,
    batch_size=train_args['batch'],
    collate_fn=correctBatch,
    num_workers=4,
)

valid_loader = DataLoader(
    dataset=valid_set,
    batch_size=recog_args['batch'],
    collate_fn=correctBatch,
    num_workers=4,
)

device = torch.device(device)
model = RoBart(device, lr = float(train_args['lr']))

if args['stage'] <= 0:
    print("training")
    print(f"# of training data:{len(train_json['token'])}")
    print(f"# of validation data:{len(valid_json['token'])}")

    min_val = 1e8
    dev_loss = []
    train_loss = []
    logging_loss = 0.0
    for e in range(train_args['epochs']):
        print(f'epoch {e + 1}: training')
        model.train()
        for n, data in enumerate(tqdm(train_loader)):
            token, mask, label = data
            # logging.warning(f'token.shape:{token.shape}')
            # logging.warning(f'token:{token}')

            # logging.warning(f'label.shape:{label.shape}')
            # logging.warning(f'label:{label}')
            token = token.to(device)
            mask = mask.to(device)
            label = label.to(device)

            loss = model(token, mask, label)

            loss /= train_args['accumgrad']
            loss.backward()
            logging_loss += loss.clone().detach().cpu()

            if ((n + 1) % train_args['accumgrad'] == 0) or ((n + 1) == len(train_loader)):
                model.optimizer.step()
                model.optimizer.zero_grad()

            if ((n + 1) % train_args['print_loss'] == 0) or (n + 1) == len(train_loader):
                logging.warning(
                    f"Training epoch :{e + 1} step:{n + 1}, training loss:{logging_loss}"
                )
                train_loss.append(logging_loss)
                logging_loss = 0.0

        train_checkpoint = dict()
        train_checkpoint["epoch"] = e + 1
        train_checkpoint["state_dict"] = model.model.state_dict()
        train_checkpoint["optimizer"] = model.optimizer.state_dict()
            
        if (not os.path.exists(f"./checkpoint/{setting}/{args['nbest']}best")):
            os.makedirs(f"./checkpoint/{setting}/{args['nbest']}best")
        torch.save(
            train_checkpoint,
            f"./checkpoint/{setting}/{args['nbest']}best/checkpoint_train_{e + 1}.pt",
        )
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            print(f'validation:')
            for n, data in enumerate(tqdm(valid_loader)):
                token, mask, label = data
                token = token.to(device)
                mask = mask.to(device)
                label = label.to(device)

                loss = model(token, mask, label)
                val_loss += loss

            dev_loss.append(val_loss)

            logging.warning(f"epoch :{e + 1}, validation_loss:{val_loss}")

            if val_loss < min_val:
                min_val = val_loss
                torch.save(
                    train_checkpoint,
                    f"./checkpoint/{setting}/{args['nbest']}best/checkpoint_train_best.pt",
                )


        save_loss = {
            "training_loss": train_loss,
            "dev_loss": dev_loss,
        }
        if not os.path.exists(f"./log/RoBart"):
            os.makedirs("./log/RoBart")
        torch.save(save_loss, f"./log/RoBart/loss.pt")

if (args['stage'] <= 1):
    print(f'Recognizing')
    recog_task = ['dev', 'test']
    dev_loader = DataLoader(
        dataset=dev_set,
        batch_size=recog_args['batch'],
        collate_fn=correctRecogBatch,
        num_workers=4,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=recog_args['batch'],
        collate_fn=correctRecogBatch,
        num_workers=4,
    )


    best_checkpoint = torch.load(f"./checkpoint/{setting}/{args['nbest']}best/checkpoint_train_best.pt")
    
    print(f"using epoch:{best_checkpoint['epoch']}")
    model.model.load_state_dict(best_checkpoint['state_dict'])

    for task in recog_task:
        model.eval()
        if (task == 'dev'):
            scoring_loader = dev_loader
        elif (task == 'test'):
            scoring_loader = test_loader
        
        recog_dict = dict()
        recog_dict['utts'] = dict()

        with torch.no_grad():        
            for i, data in enumerate(tqdm(scoring_loader)):
                temp_dict = dict()
                token, mask, ref = data
                token = token.to(device)
                mask = mask.to(device)

                output = model.recognize(token, mask)
                
                output = output.squeeze(0).tolist()

                hyp_token = model.tokenizer.convert_ids_to_tokens(output)
                hyp_token = [str(h) for h in hyp_token if h not in ['[CLS]', '[SEP]', '[PAD]']]
                
                ref = [t for t in ref[0]]
                logging.warning(f'hyp:{hyp_token}')
                logging.warning(f'ref:{ref}')

                recog_dict['utts'][f'{task}_{i}'] = dict()
                recog_dict['utts'][f'{task}_{i}']['output'] = {
                    'recog_text': " ".join(hyp_token),
                    'ref_token': " ".join(ref)
                }

        if (not os.path.exists(
            f"./data/{args['dataset']}/{setting}/{task}//{args['nbest']}best")
        ):
            os.makedirs(f"./data/{args['dataset']}/{setting}/{task}/{args['nbest']}best")
        with open(
            f"./data/{args['dataset']}/{setting}/{task}/{args['nbest']}best/rescore_data.json",
            'w'
        ) as fw:
            json.dump(recog_dict, fw, ensure_ascii = False, indent = 4)
print(f'Finish')


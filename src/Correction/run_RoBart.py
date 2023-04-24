import os
from tqdm import tqdm
import random
import yaml
import logging
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.BartForCorrection.RoBart import RoBart
from utils.Datasets import get_dataset
from utils.CollateFunc import trainBatch, recogBatch
from utils.PrepareModel import prepare_model
from transformers import BertTokenizer
from src_utils.LoadConfig import load_config
from jiwer import wer
from torch.optim import AdamW

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
config = f"./config/Bart.yaml"

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
if (args['dataset'] == 'old_aishell'):
    setting = ""

print(f"Prepare data")
print(f'Load json file')

if (args['dataset'] in ['aishell2']):
    dev = 'dev_ios'
    test = ['test_ios', 'test_mic', 'test_android']

# test_path = f"../../data/{args['dataset']}/data/test/{setting}/data.json"

model , tokenizer = prepare_model(args['dataset'])
optimizer = AdamW(model.parameters(), lr = 1e-5)

# model = RoBart(device, lr = float(train_args['lr']))

if args['stage'] <= 0:
    print("training")
    train_path = f"../../data/{args['dataset']}/data/{setting}/train/data.json"
    dev_path = f"../../data/{args['dataset']}/data/{setting}/dev/data.json"

    with open(train_path) as f,\
     open(dev_path) as d:
        train_json = json.load(f)
        dev_json = json.load(d)

    print(f'Create Dataset & DataLoader')
    train_set = get_dataset(train_json, tokenizer,topk = args['nbest'], for_train = True)
    valid_set = get_dataset(dev_json, tokenizer, topk = args['nbest'], for_train = True)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_args['train_batch'],
        collate_fn=trainBatch,
        num_workers=4,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=train_args['valid_batch'],
        collate_fn=trainBatch,
        num_workers=4,
    )

    min_val = 1e8
    min_cer = 100
    dev_loss = []
    train_loss = []
    logging_loss = 0.0
    model = model.to(device)
    for e in range(train_args['epoch']):
        print(f'epoch {e + 1}: training')
        model.train()
        for n, data in enumerate(tqdm(train_loader)):
            data = {k: v.to(device)  for k, v in data.items()}
            loss = model(**data, return_dict = True).loss

            loss /= train_args['accumgrad']
            loss.backward()
            logging_loss += loss.clone().detach().cpu()

            if ((n + 1) % train_args['accumgrad'] == 0) or ((n + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            if ((n + 1) % train_args['print_loss'] == 0) or (n + 1) == len(train_loader):
                logging.warning(
                    f"Training epoch :{e + 1} step:{n + 1}, training loss:{logging_loss}"
                )
                train_loss.append(logging_loss)
                logging_loss = 0.0

        train_checkpoint = dict()
        train_checkpoint["epoch"] = e + 1
        train_checkpoint["state_dict"] = model.state_dict()
        train_checkpoint["optimizer"] = optimizer.state_dict()
            
        if (not os.path.exists(f"./checkpoint/{setting}/{args['nbest']}best")):
            os.makedirs(f"./checkpoint/{setting}/{args['nbest']}best")
        torch.save(
            train_checkpoint,
            f"./checkpoint/{setting}/{args['nbest']}best/checkpoint_train_{e + 1}.pt",
        )
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            refs = []
            hyps = []
            print(f'validation:')
            for n, data in enumerate(tqdm(valid_loader)):
                data = {k:v.to(device) for k, v in data.items()}

                loss = model(**data, return_dict = True).loss
                hyp = model.generate(input_ids = data["input_ids"], attention_mask = data["attention_mask"])
                hyp = tokenizer.batch_decode(hyp, skip_special_tokens = True)
                labels = data["labels"].clone().cpu()
                ref = np.where(labels != -100, labels, tokenizer.pad_token_id)
                ref = tokenizer.batch_decode(ref, skip_special_tokens = True)

                for h, r in zip(hyp, ref):
                    hyps.append(h)
                    refs.append(r)

                val_loss += loss
            cer = wer(refs, hyps)

            dev_loss.append(val_loss)

            logging.warning(f"epoch :{e + 1}, validation_loss:{val_loss}")
            logging.warning(f'epoch :{e + 1}, validation cer: {cer}')

            if cer < min_cer:
                min_val = val_loss
                torch.save(
                    train_checkpoint,
                    f"./checkpoint/{setting}/{args['nbest']}best/checkpoint_train_best.pt",
                )
            logging.warning(f'hyp:{hyps[0]}')
            logging.warning(f'ref:{refs[0]}')


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

    if (args['dataset'] in ['aishell2']):
        recog_task = ['dev_ios', 'test_mic', 'test_ios', 'test_android']

    best_checkpoint = torch.load(f"./checkpoint/{setting}/{args['nbest']}best/checkpoint_train_best.pt")
    
    print(f"using epoch:{best_checkpoint['epoch']}")
    model.load_state_dict(best_checkpoint['state_dict'])
    model = model.to(device)

    for task in recog_task:
        print(f'task:{task}')
        model.eval()
        with open(f"../../data/{args['dataset']}/data/{setting}/{task}/data.json") as f:
            data_json = json.load(f)
        
        dataset = get_dataset(data_json, tokenizer, topk = 1, for_train = False)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=recog_args['batch'],
            collate_fn=recogBatch,
            num_workers=4,
        )

        recog_dict = dict()
        hyp_cal = []
        ref_cal = []

        with torch.no_grad():        
            for i, data in enumerate(tqdm(dataloader)):
                temp_dict = dict()

                names = data['name']
                input_ids = data['input_ids'].to(device)
                logging.warning(f'{input_ids}')
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels']
                
                output = model.generate(input_ids, attention_mask = attention_mask,num_beams = 3, min_length = 0 ,max_length = 50)
                logging.warning(f'output:{output}')

                labels = np.where(labels != -100, labels, tokenizer.pad_token)

                hyps = tokenizer.batch_decode(output, skip_special_tokens = True)
                refs = tokenizer.batch_decode(labels, skip_sprcial_tokens = True)

                for name, hyp, ref in zip(names, hyps, refs):
                    hyp= hyp.split()[1:-1]
                    hyp = " ".join(hyp)

                    recog_dict[name] = {
                        "hyp": hyp,
                        "ref": ref
                    }
                    hyp_cal.append(hyp)
                    ref_cal.append(ref)

            print(hyp_cal[-1])
            print(ref_cal[-1])
            print(f"WER: {wer(ref_cal, hyp_cal)}")

        if (not os.path.exists(
            f"./data/{args['dataset']}/{setting}/{task}/{args['nbest']}best")
        ):
            os.makedirs(f"./data/{args['dataset']}/{setting}/{task}/{args['nbest']}best")

        with open(
            f"./data/{args['dataset']}/{setting}/{task}/{args['nbest']}best/correct_data.json",
            'w'
        ) as fw:
            json.dump(recog_dict, fw, ensure_ascii = False, indent = 4)

print(f'Finish')


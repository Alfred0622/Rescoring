import os
from secrets import token_urlsafe
from tqdm import tqdm
import random
import json
import yaml
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from models.BertForRescoring.GPT2ForRescoring import CLMRescorer
from transformers import BertTokenizerFast
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

if (not os.path.exists("./log/clm")):
    os.makedirs(f"./log/clm/")

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/clm/train.log",
    filemode="w",
    format=FORMAT,
)

from utils.Datasets import (
    causalLMDataset,
    causalLMDRecogDataset
)
from utils.CollateFunc import(
    lmBatch,
    lmRecogBatch
)
from utils.LoadConfig import load_config


random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

"""Basic setting"""
# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

config = f"./config/clm.yaml"
args, train_args, recog_args = load_config(config)

setting = 'withLM' if args['withLM'] else 'noLM'
print(f'Use:{setting}')

print('prepare Data')
with open(train_args['train_json'], 'r') as train, \
     open(train_args['dev_json'], 'r') as dev, \
     open(train_args['test_json'], 'r') as test:
     
     train_json = json.load(train)
     dev_json = json.load(dev)
     test_json = json.load(test)

# set Dataset
train_dataset = causalLMDataset(
    train_json,
    nbest = args['nbest']
)

valid_dataset = causalLMDataset(
    dev_json,
    nbest = args['nbest']
)

dev_dataset = causalLMDRecogDataset(
    dev_json,
    nbest = args['nbest']
)

test_dataset = causalLMDRecogDataset(
    test_json,
    nbest = args['nbest']
)

# Dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size = train_args['train_batch'],
    collate_fn = lmBatch,
    num_workers = 4
)

valid_loader = DataLoader(
    dev_dataset,
    batch_size = train_args['valid_batch'],
    collate_fn = lmBatch,
    num_workers = 4
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size = recog_args['batch'],
    collate_fn = lmRecogBatch,
    num_workers = 4
)

test_loader = DataLoader(
    test_dataset,
    batch_size = recog_args['batch'],
    collate_fn = lmRecogBatch,
    num_workers = 4
)

# set model
model = CLMRescorer(
    device = device,
    lr = train_args['lr'],
    mode = train_args['mode'],
)
if args['stage'] <= 1 and args['stop_stage']>= 1:
    print('Training...')
    min_valid_loss = 1e8
    for e in range(train_args['epoch']):
        model.train()
        accum_loss = 0.0
        logging_loss = 0.0
        for i, data in enumerate(tqdm(train_loader)):
        
            _, _,ref_token, ref_mask,  _, _ = data
            ref_token = ref_token.to(device)
            ref_mask = ref_mask.to(device)

            loss = model(input_ids = ref_token, attention_mask = ref_mask,labels = ref_token)
            loss = loss /  train_args['accumgrad']
            loss.backward()
            logging_loss += loss.clone().detach().cpu()

            if ((i + 1) % train_args['accumgrad'] == 0 or (i + 1) == len(train_loader)):
                model.optimizer.step()
                model.optimizer.zero_grad()

            if (
            (
                (i + 1) % (train_args['print_loss']) == 0
            ) or 
                ( (i + 1) == len(train_loader))
            ):
                logging.warning(f'train_step:{i + 1}: loss:{logging_loss}')
                logging_loss = 0.0
            
        checkpoint = {
            'epoch': e + 1,
            'state_dict': model.model.state_dict(),
            'optimizer': model.optimizer.state_dict()
        }
        # Save state dict
        if (not os.path.exists(f'./checkpoint/clm/{setting}')):
            os.makedirs(f'./checkpoint/clm/{setting}')

        torch.save(checkpoint, f'./checkpoint/clm/{setting}/checkpoint_train_{e + 1}.pt')
        
        model.eval()
        valid_loss = 0.0
        c = 0
        s = 0
        d = 0
        i = 0
        for i, data in enumerate(tqdm(valid_loader)):
            _, _, ref_token, ref_mask, _, _ = data

            ref_token = ref_token.to(device)
            ref_mask = ref_mask.to(device)

            loss = model(input_ids = ref_token, attention_mask = ref_mask,labels = ref_token)
            
            valid_loss += loss.clone().detach().cpu()
        logging.warning(f'epoch: {e + 1}, valid loss:{valid_loss}')
        if (valid_loss < min_valid_loss):
            min_valid_loss = valid_loss

            torch.save(
                checkpoint,
                f'./checkpoint/clm/{setting}/checkpoint_train_best.pt'
            )
        
if args['stage']<= 2 and args['stop_stage']>= 2:
    
    checkpoint = torch.load(f'./checkpoint/clm/{setting}/checkpoint_train_best.pt')
    
    best_epoch = checkpoint['epoch']
    model.model.load_state_dict(checkpoint['state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer'])

    print(f'Scoring, using best_epoch:{best_epoch}')
    recog_set = ['dev', 'test']
    
    for task in recog_set:
        recog_dict = list()
        print(f'scoring:{task}')
        if (task == 'dev'):
            scoring_loader = dev_loader
        elif (task == 'test'):
            scoring_loader = test_loader
        for i, data in enumerate(tqdm(scoring_loader)):
            token, mask, err, first_score, text, ref = data

            token = token.to(device)
            mask = mask.to(device)
            scores = model.recognize(token, attention_mask = mask)
            scores = scores.tolist()

            recog_dict.append(
                {
                    'text': text,
                    'first_score': first_score.tolist(),
                    'rescore': scores,
                    'cer': err.tolist(),
                    'ref': ref
                }
            )
        
        save_path = f'./data/aishell/{task}/CLM/{setting}'
        if (not os.path.exists(save_path)):
            os.makedirs(save_path)
        with open(f'{save_path}/recog_data.json', 'w') as f:
            json.dump(recog_dict, f, ensure_ascii = False, indent = 4)

if args['stage']<= 3 and args['stop_stage']>= 3:
    print(f'find best weight')
    recog_set = ['dev', 'test']
    
    load_path = f'./data/aishell/dev/CLM/{setting}'
    
    min_cer = 100
    best_weight = 0.0
    with open(f'{load_path}/recog_data.json', 'r') as f:
        dev_json = json.load(f)
        for w in tqdm(range(101)):
            weight = w * 0.01
            c = 0
            s = 0
            d = 0
            i = 0
            for n, data in enumerate(dev_json):
                first_score = torch.tensor(data['first_score'])
                rescore = torch.tensor(data['rescore'])

                total_score = (1 - weight) * first_score + (weight) * rescore

                max_index = torch.argmax(total_score).item()

                c += data['cer'][max_index][0]
                s += data['cer'][max_index][1]
                d += data['cer'][max_index][2]
                i += data['cer'][max_index][3]
            cer = (s + d + i) / (c + s + d)

            logging.warning(f'weight:{weight}, CER:{cer}')

            if (cer < min_cer):
                print(f'weight:{weight}, cer:{cer}')
                min_cer = cer
                best_weight = weight
    
    for task in recog_set:
        load_path = f'./data/aishell/{task}/CLM/{setting}'
        with open(f'{load_path}/recog_data.json', 'r') as f:
            scoring_data = json.load(f)
            rescore_dict = dict()
            rescore_dict['utts'] = dict()

            c = 0
            s = 0
            d = 0
            i = 0

            for n, data in enumerate(tqdm(scoring_data)):
                first_score = torch.tensor(data['first_score'])
                rescore = torch.tensor(data['rescore'])

                total_score = (1 - best_weight) * first_score + best_weight * rescore

                max_index = torch.argmax(total_score).item()

                c += data['cer'][max_index][0]
                s += data['cer'][max_index][1]
                d += data['cer'][max_index][2]
                i += data['cer'][max_index][3]

                text = data['text'][max_index].split()
                ref = data['ref']
                rescore_dict['utts'][f'{task}_{n}'] = dict()
                rescore_dict['utts'][f'{task}_{n}']['output'] = {
                    "rec_token" : " ".join(text),
                    "ref_token" : " ".join(ref),
                    "first_score" : data['first_score'],
                    "rescore" : data['rescore'],
                    "weighted_score": total_score.tolist()
                }
            # print(f'correct:{c}\n substitution:{s}\n deletion:{d}\n insertion:{i}')
            cer = (s + i + d) / (c + s + d)
            print(f"{setting} {task}: {cer}")
            with open(f'{load_path}/rescore_data.json', 'w') as fw:
                json.dump(
                    rescore_dict, fw, ensure_ascii = False, indent = 4
                )
print("Finish")

import os
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

from utils.Datasets import (
    correctDataset
)
from utils.CollateFunc import(
    correctBatch,
    correctRecogBatch
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
train_dataset = correctDataset(
    train_json,
    nbest = args['nbest']
)

dev_dataset = correctDataset(
    dev_json,
    nbest = args['nbest']
)

test_dataset = correctDataset(
    test_json,
    nbest = args['nbest']
)

train_loader = DataLoader(
    train_dataset,
    batch_size = train_args['train_batch'],
    collate_fn = correctBatch,
    pin_memory = True,
    num_workers = 4
)

valid_loader = DataLoader(
    dev_dataset,
    batch_size = train_args['valid_batch'],
    collate_fn = correctBatch,
    pin_memory = True,
    num_workers = 4
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size = recog_args['batch'],
    collate_fn = correctRecogBatch,
    pin_memory = True,
    num_workers = 4
)

test_loader = DataLoader(
    dev_dataset,
    batch_size = recog_args['batch'],
    collate_fn = correctRecogBatch,
    pin_memory = True,
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
        
            token, ref_token = data
            token = token.to(device)
            ref_token = ref_token.to(device)

            loss = model(input_ids = token, labels = ref_token) / train_args['accumgrad']
            loss.backward()
            logging_loss += loss.clone().detach().cpu()

            if (i % train_args['accumgrad'] == 0 or i == len(train_loader)):
                model.optimizer.step()
                model.optimizer.zero_grad()

            if (i % train_args['print_loss'] == 0 or i == len(train_loader)):
                logging.warning(f'train_step:{i}: loss:{logging_loss}')
                logging_loss = 0.0
            
        checkpoint = {
            'epoch': e + 1,
            'state_dict': model.model.state_dict(),
            'optimizer': model.optimizer.state_dict()
        }
        # Save state dict
        if (not os.path.exists('./checkpoint/clm')):
            os.makedirs('./checkpoint/clm')

        torch.save(checkpoint, f'./checkpoint/clm/checkpoint_train_{e + 1}.pt')
        
        model.eval()
        valid_loss = 0.0
        for i, data in valid_loader:
            token, ref_token = data

            valid_loss += model(token, ref_token)
        
        logging.warning(f'epoch: {e + 1}, valid loss:{valid_loss}')
        if (dev_loss < min_valid_loss):
            min_valid_loss = valid_loss

            torch.save(
                checkpoint,
                './checkpoint/clm/checkpoint_train_best.pt'
            )
        
if args['stage']<= 2 and args['stop_stage']>= 2:
    print(f'Scoring')
    recog_set = ['dev', 'test']
    
    recog_dict = list()
    for task in recog_set:
        if (task == 'dev'):
            scoring_loader = dev_loader
        elif (task == 'test'):
            scoring_loader = test_loader
        for i, data in enumerate(scoring_loader):
            token, err, text, ref, first_score = data
            scores = model.recognize(token)
            scores = scores.tolist()

            recog_dict.append(
                {
                    'text': text,
                    'first_score': first_score,
                    'rescore': scores.tolist,
                    'cer': err.tolist(),
                    'ref': ref
                }
            )
        
        save_path = f'./data/aishell/{task}/CLM/{setting}'
        if (not os.path.exists(save_path)):
            os.makedirs(save_path)
        with open(f'{save_path}/rescore_data.json', 'w') as f:
            json.dump(recog_dict, f, ensure_ascii = False, indent = 4)

if args['stage']>= 3 and args['stop_stage']<= 3:
    print(f'find best weight')
    recog_set = ['dev', 'test']
    
    load_path = f'./data/aishell/dev/CLM/{setting}'

    with open(f'{load_path}/rescore_data.json', 'r') as f:
        dev_json = json.load(f)
        min_cer = 100
        best_weight = 0.0
        for w in range(101):
            weight = w * 0.01
            c = 0
            s = 0
            d = 0
            i = 0
            for n, data in enumerate(tqdm(dev_json)):
                first_score = torch.tensor(data['first_score'])
                rescore = torch.tensor(data['rescore'])

                total_score = (1 - weight) * first_score + (weight) * rescore

                max_index = torch.argmax(total_score).item()

                c += data['cer'][max_index][0]
                s += data['cer'][max_index][1]
                d += data['cer'][max_index][2]
                i += data['cer'][max_index][3]
            cer = (s + d + i) / (c + s + d)

            if (cer < min_cer):
                print(f'weight:{weight}, cer:{cer}')
                min_cer = cer
                best_weight = weight
    
    for task in recog_set:
        load_path = load_path = f'./data/aishell/{task}/CLM/{setting}'
        with open(f'{load_path}/rescore_data.json', 'r') as f:
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

                total_score = (1 - weight) * first_score + (weight) * rescore

                max_index = torch.argmax(total_score).item()

                c += data['cer'][max_index][0]
                s += data['cer'][max_index][1]
                d += data['cer'][max_index][2]
                i += data['cer'][max_index][3]

                text = data['text'][max_index].split()
                ref = data['ref'].split()
                rescore_dict['utts'][f'{task}_{n}'] = dict()
                rescore_dict['utts'][f'{task}_{n}']['output'] = {
                    "rec_token" : " ".join(text),
                    "ref_token" : " ".join(ref),
                    "first_score" : data['first_score'],
                    "rescore" : data['rescore'],
                    "weighted_score": total_score.tolist()
                }
            cer = (s + i + d) / (c + s + d)
            print("{} {}: {:.2f}".format(setting, task, cer))
print("Finish")

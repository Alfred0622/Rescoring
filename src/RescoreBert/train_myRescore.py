import os
import sys
sys.path.append("../")
import json
import logging
import random
import torch
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.nn.functional import log_softmax
from utils.LoadConfig import load_config
from utils.Datasets import prepare_myDataset
from utils.CollateFunc import myCollate
from utils.PrepareModel import prepare_myModel
from utils.cal_score import get_sentence_score

config_name = '/mnt/disk6/Alfred/Rescoring/src/RescoreBert/config/myRescoreBert.yaml'

os.environ["TOKENIZERS_PARALLELISM"] = "true"

args, train_args, recog_args = load_config(config_name)

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

if (torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

setting = 'withLM' if (args['withLM']) else 'noLM'

logging_path = Path(f"./log/MyRescoreBert/{args['dataset']}/{setting}/batch{train_args['batch_size']}_lr{train_args['lr']}_lstm{train_args['lstm_embedding']}")
logging_path.mkdir(exist_ok=True, parents = True)

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"{logging_path}/train.log",
    filemode="w",
    format=FORMAT,
)

if (args['dataset'] in ['aishell2']):
    dev_set = 'dev_ios'
elif (args['dataset'] in ["librispeech"]):
    dev_set = 'valid'
else:
    dev_set = 'dev'

model, bert_tokenizer, gpt2, gpt_tokenizer = prepare_myModel(args['dataset'], lstm_dim=train_args['lstm_embedding'], device = device)
bos = gpt_tokenizer.bos_token_id if (gpt_tokenizer.bos_token_id is not None) else gpt_tokenizer.cls_token_id
eos = gpt_tokenizer.eos_token_id if (gpt_tokenizer.eos_token_id is not None) else gpt_tokenizer.sep_token_id
pad = gpt_tokenizer.pad_token_id

gpt_checkpoint = torch.load("/mnt/disk6/Alfred/Rescoring/src/RescoreBert/checkpoint/aishell/CLM_lr1e-5/checkpoint-14100/pytorch_model.bin")
gpt2.load_state_dict(gpt_checkpoint)

with open(f"../../data/{args['dataset']}/data/{setting}/train/data.json") as f, \
     open(f"../../data/{args['dataset']}/data/{setting}/{dev_set}/data.json") as dev:
    train_json = json.load(f)
    valid_json = json.load(dev)

train_dataset = prepare_myDataset(data_json = train_json, bert_tokenizer = bert_tokenizer, gpt_tokenizer = gpt_tokenizer, topk = 50)
valid_dataset = prepare_myDataset(data_json = valid_json, bert_tokenizer = bert_tokenizer, gpt_tokenizer = gpt_tokenizer, topk = 50)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = train_args['batch_size'],
    collate_fn = myCollate,
    num_workers = 8,
    pin_memory = True,
    prefetch_factor = 32
)

valid_loader = DataLoader(
    dataset = valid_dataset,
    batch_size = train_args['batch_size'],
    collate_fn = myCollate,
    num_workers = 8,
    pin_memory = True,
    prefetch_factor = 32
)

optimizer = Adam(
    list(model.parameters()) + list(gpt2.parameters()),
    lr = float(train_args['lr'])
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr = float(train_args['lr']), 
    steps_per_epoch = len(train_loader), 
    epochs = train_args['epoch'],
    pct_start = 0.15
)

logging_loss = 0.0
min_valid_loss = 1e8

optimizer.zero_grad()

for e in range(int(train_args['epoch'])):

    model.train()
    gpt2.train()

    if (e > train_args['freeze_gpt_epoch']):
        for param in gpt2.parameters():
            param.requires_grad = False
    '''
        Training 
    '''
    for step, data in enumerate(tqdm(train_loader, ncols = 100)):
        bert_ids = data['bert_ids'].to(device)
        gpt_ids = data['gpt_ids'].to(device)

        bert_mask = data['bert_mask'].to(device)
        gpt_mask = data['gpt_mask'].to(device)

        am_scores = data['am_score'].to(device)
        ctc_scores = data['ctc_score'].to(device)


        gpt_output = gpt2(
            input_ids = gpt_ids,
            attention_mask = gpt_mask
        ).logits

        gpt_scores = log_softmax(gpt_output, dim = -1) # label
        gpt_scores = get_sentence_score(gpt_scores, gpt_ids, bos, eos, pad)

        loss = model(
            bert_ids,
            bert_mask,
            gpt_scores,
            am_scores,
            ctc_scores,
        ).loss

        loss = loss / train_args['accumgrad']

        loss.backward()

        logging_loss += loss.item()

        if ( (step + 1) % int(train_args['accumgrad']) == 0):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if ( (step + 1 % int(train_args['print_loss'])) == 0 or step + 1 == len(train_loader)):
            logging.warning(f'Epoch:{e + 1} - Step:{step + 1} -- train_loss = {logging_loss}')
            logging_loss = 0.0
        
    """
        Validation
    """
    valid_loss = 0.0
    model.eval()
    gpt2.eval()
    with torch.no_grad():
        for step, data in enumerate(tqdm(valid_loader, ncols = 100)):
            bert_ids = data['bert_ids'].to(device)
            gpt_ids = data['gpt_ids'].to(device)

            bert_mask = data['bert_mask'].to(device)
            gpt_mask = data['gpt_mask'].to(device)

            am_scores = data['am_score'].to(device)
            ctc_scores = data['ctc_score'].to(device)

            gpt_output = gpt2(
                input_ids = gpt_ids,
                attention_mask = gpt_mask
            ).logits

            gpt_scores = log_softmax(gpt_output, dim = -1) # label
            gpt_scores = get_sentence_score(gpt_scores, gpt_ids, bos, eos, pad)

            loss = model(
                bert_ids,
                bert_mask,
                gpt_scores,
                am_scores,
                ctc_scores,
            ).loss

            valid_loss += loss.item()
        
        valid_loss = valid_loss / len(valid_loader)
        logging.warning(f'Epoch:{e + 1}, Validation loss:{valid_loss}')

    checkpoint = {
        "model": model.state_dict(),
        "gpt2": gpt2.state_dict()
    }
    
    checkpoint_path = Path(f"./checkpoint/{args['dataset']}/MyRescoreBert/{setting}")
    checkpoint_path.mkdir(parents = True, exist_ok = True)

    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_{e + 1}.pt")

    if (valid_loss < min_valid_loss):
        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best.pt")
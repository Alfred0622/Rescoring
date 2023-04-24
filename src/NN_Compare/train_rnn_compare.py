import os
import sys
sys.path.insert(0, f'..')
import json
import torch
import logging
from model.RNN_Rerank import RNN_Reranker
from src_utils.LoadConfig import load_config
from src_utils.get_dict import get_vocab
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.utils.Datasets import get_Dataset
from model.utils.CollateFunc import trainBatch
from pathlib import Path

config = f'./config/RNN_Embed_Only.yaml'

args, train_args, _ = load_config(config)
setting = "withLM" if (args['withLM']) else "noLM"

vocab_dict = f"./data/{args['dataset']}/lang_char/vocab.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if (not os.path.exists("./log")):
    os.makedirs("./log")

with open(vocab_dict, 'r') as f:
    dict_file = f.readlines()

vocab_dict = get_vocab(dict_file)

print(f'vocab_size:{len(vocab_dict)}')
FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"

logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/{setting}_RNN_compare.log",
    filemode="w",
    format=FORMAT,
)

if (train_args["hidden_dim"] <= train_args["output_dim"]):
    output_dim = 0
else:
    output_dim = train_args['output_dim']

model = RNN_Reranker(
    vocab_size = len(vocab_dict),
    hidden_dim = train_args["hidden_dim"],
    num_layers = 1,
    output_dim = output_dim,
    device = device,
    lr = float(train_args['lr']),
    add_additional_feat = args['add_additional_feat'],
    add_am_lm_score = args['add_am_lm_score'],
)

with open(f"../../data/{args['dataset']}/token/{setting}/train/token.json") as f, \
     open(f"../../data/{args['dataset']}/token/{setting}/dev/token.json") as v:
    train_json = json.load(f)
    valid_json = json.load(v)

train_dataset = get_Dataset(train_json)
valid_dataset = get_Dataset(valid_json)

print(f'size of train:{len(train_dataset)}')

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size=train_args['train_batch'],
    collate_fn = trainBatch
)

valid_loader = DataLoader(
    dataset = valid_dataset,
    batch_size=train_args['valid_batch'],
    collate_fn = trainBatch
)

min_loss = 1e8
for e in range(train_args['train_epoch']):
    model.optimizer.zero_grad()
    model.train()
    logging_loss = 0.0
    for step, data in enumerate(tqdm(train_loader)):
        
        input_1 = data['input_1'].to(device)
        input_2 = data['input_2'].to(device)
        
        am_score = data['am_score'].to(device)
        lm_score = data['lm_score'].to(device)
        ctc_score = data['ctc_score'].to(device)

        labels = data['labels'].to(device)

        loss = model(
            input_1,
            input_2,
            am_score = am_score,
            ctc_score = ctc_score,
            lm_score = lm_score,
            labels = labels
        ).loss

        loss = loss/train_args['accum_grad']

        logging_loss += loss.item()

        loss.backward()
        if ((step + 1) % train_args['accum_grad'] == 0):
            model.optimizer.step()
            model.optimizer.zero_grad()
        if ((step + 1) % train_args['logging_loss'] == 0 or (step + 1 == len(train_loader))):
            logging.warning(f'epoch : {e + 1}, step:{step + 1}: training_loss:{logging_loss}')
            logging_loss = 0.0
    
    checkpoint_path = Path(f"./checkpoint/{args['dataset']}/{setting}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": e + 1,
        "state_dict": model.state_dict()
    }

    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_{e+1}.pt")
    
    # Eval
    model.eval()
    with torch.no_grad():
        eval_loss = 0.0
        for data in tqdm(valid_loader):
            
            input_1 = data['input_1'].to(device)
            input_2 = data['input_2'].to(device)
            # logging.warning(f'input_1:{input_1.shape}')
            # logging.warning(f'input_2:{input_2.shape}')
            
            am_score = data['am_score'].to(device)
            lm_score = data['lm_score'].to(device)
            ctc_score = data['ctc_score'].to(device)

            labels = data['labels'].to(device)

            loss = model(
                input_1,
                input_2,
                am_score = am_score,
                ctc_score = ctc_score,
                lm_score = lm_score,
                labels = labels
            ).loss

            loss = loss / train_args['accum_grad']

            eval_loss += loss.item()

        logging.warning(f'training epoch: {e + 1}, validation loss: {eval_loss}')

        if (eval_loss < min_loss):
            min_loss = eval_loss
            torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best.pt")




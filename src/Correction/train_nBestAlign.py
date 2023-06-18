import os

from tqdm import tqdm
import random
import json
import yaml
import logging
import torch
from torch.utils.data import DataLoader
from models.nBestAligner.nBestTransformer import nBestAlignBart
from transformers import BertTokenizer
from utils.PrepareModel import prepare_model
from src_utils.LoadConfig import load_config
from utils.Datasets import  get_dataset
from utils.CollateFunc import nBestAlignBatch
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
import wandb
from pathlib import Path

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

config = f"./config/nBestAlign.yaml"

args, train_args, recog_args = load_config(config)

setting = 'withLM' if args['withLM'] else 'noLM'
nbest = int(args['nbest'])

model, tokenizer = prepare_model(dataset = args['dataset'])

if (not os.path.exists(f"./log/nBestAlign/{args['dataset']}/{setting}")):
    os.makedirs(f"./log/nBestAlign/{args['dataset']}/{setting}")

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/nBestAlign/{args['dataset']}/{setting}/{args['nbest']}Align_train.log",
    filemode="w",
    format=FORMAT,
)

if (args['dataset'] in ['aishell', 'tedlium2', 'csj']):
    valid = 'dev'
elif (args['dataset'] in ['aishell2']):
    valid = 'dev_ios'
elif (args['dataset'] in ['librispeech']):
    valid = 'valid'

if ('WANDB_MODE' in os.environ.keys() and os.environ['WANDB_MODE'] == 'disabled'):
    fetch_num = 500
else:
    fetch_num = -1

if __name__ == "__main__":
    train_path = f"../../data/{args['dataset']}/data/{setting}/train/data.json"
    dev_path = f"../../data/{args['dataset']}/data/{setting}/{valid}/data.json"
    
    print(f"Prepare data")
    with open(train_path) as f,\
         open(dev_path) as d:
        train_json = json.load(f)
        dev_json = json.load(d)

    train_set = get_dataset(
            train_json, 
            dataset = args['dataset'],
            tokenizer = tokenizer, 
            data_type = 'align', 
            topk = int(args['nbest']), 
            sep_token = train_args['sep_token'],
            fetch_num  = fetch_num
        )
    dev_set = get_dataset(
        dev_json, 
        dataset = args['dataset'],
        tokenizer = tokenizer, 
        data_type = 'align', 
        topk = int(args['nbest']), 
        sep_token = train_args['sep_token'],
        fetch_num  = fetch_num
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_args["train_batch"],
        collate_fn=nBestAlignBatch,
        # num_workers=4,
        shuffle=False
    )

    dev_loader = DataLoader(
        dataset=dev_set,
        batch_size=recog_args['batch'],
        collate_fn=nBestAlignBatch,
        # num_workers=4,
        shuffle = False
    )

    logging.warning(f"device:{device}")
    device = torch.device(device)

    model = nBestAlignBart(
            args,
            train_args,
    ).to(device)

    if (train_args['from_pretrain']):
        pretrain_name = "Pretrain"
    else:
        pretrain_name = "Scratch"

    optimizer = AdamW(model.parameters(), lr = float(train_args['lr']))
    scheduler = OneCycleLR(
        optimizer,
        epochs = int(train_args['epoch']),
        steps_per_epoch = len(train_loader),
        pct_start = 0.01,
        anneal_strategy='linear',
        max_lr = float(train_args['lr'])
    )
    print(f"training")

    min_val = 1e8


    config = {
        "args": args,
        "train_args": train_args,
        "Bart_config": model.model.config
        if (torch.cuda.device_count() <= 1)
        else model.module.model.config,
        "optimizer": optimizer
    }

    run_name = f"{args['dataset']}, {setting} : {args['nbest']}-Align"
    wandb.init(
        project=f"NBestBert_{args['dataset']}_{setting}", config=config, name=run_name
    )
    wandb.watch(model)


    for e in range(train_args['epoch']):
        model.train()

        logging_loss = 0.0
        epoch_loss = 0.0
        step = 0
        data_count = 0
        optimizer.zero_grad()
        for n, data in enumerate(tqdm(train_loader, ncols = 100)):
            # logging.warning(f'token.shape:{token.shape}')
            token = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            label = data['labels'].to(device)

            loss = model(token, mask, label)

            loss /= train_args["accumgrad"]
            loss.backward()
            data_count += 1
            logging_loss += loss.item()
            epoch_loss += loss.item()

            if ((n + 1) % train_args["accumgrad"] == 0) or ((n + 1) == len(train_loader)):
                optimizer.step()
                scheduler.step()                
                optimizer.zero_grad()
                step += 1

            if (step > 0 and step % train_args["print_loss"] == 0 or (n + 1) == len(train_loader)):
                logging.warning(
                    f"Training epoch :{e + 1} step:{n + 1}, training loss:{logging_loss / data_count}"
                )
                wandb.log(
                    {"loss": logging_loss / data_count},
                    step = e * len(train_loader) + n
                )

                logging_loss = 0.0
                step = 0
                data_count = 0
            
        checkpoint = {
            'epoch': e + 1,
            'checkpoint': model.state_dict()
        }


        checkpoint_path = Path(f"./checkpoint/{args['dataset']}/{args['nbest']}Align/{setting}")
        checkpoint_path.mkdir(parents = True, exist_ok = True)
        torch.save(
                checkpoint,
                f"{checkpoint_path}/checkpoint_train_{e+1}.pt",
        )
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for n, data in enumerate(tqdm(dev_loader, ncols = 100)):
                token = data['input_ids'].to(device)
                mask = data['attention_mask'].to(device)
                label = data['labels'].to(device)

                loss = model(token, mask, label)
                val_loss += loss

            val_loss = val_loss / len(dev_loader)

            logging.warning(f"epoch :{e + 1}, validation_loss:{val_loss}")
            wandb.log(
                {
                    "train_epoch_loss": epoch_loss / len(train_loader),
                    'val_loss':val_loss,
                    "epoch": e + 1
                },
                step = (e + 1) * len(train_loader)
            )

            if val_loss < min_val:
                min_val = val_loss
                torch.save(
                    checkpoint,
                    f"{checkpoint_path}/checkpoint_valBest.pt",
                )
            
        
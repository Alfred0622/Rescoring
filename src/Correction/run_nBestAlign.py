import os
from tqdm import tqdm
import random
import json
import yaml
import logging
import torch
from torch.utils.data import DataLoader
from models.nBestAligner.nBestTransformer import nBestTransformer
from transformers import BertTokenizer
from utils.PrepareModel import prepare_model
from src_utils.LoadConfig import load_config
from utils.Datasets import  get_dataset
from utils.CollateFunc import nBestAlignBatch
from torch.optim.lr_scheduler import OneCycleLR

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

config = f"./config/nBestAlign.yaml"

args, train_args, recog_args = load_config(config)

setting = 'withLM' if args['withLM'] else 'noLM'
nbest = int(args['nbest'])

training_mode = train_args["mode"]
model_name = train_args["model_name"]

model, tokenizer = prepare_model(dataset = args['dataset'])

if (not os.path.exists(f"./log/nBestAlign")):
    os.makedirs(f"./log/nBestAlign")

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/nBestAlign/{training_mode}_{model_name}_train.log",
    filemode="w",
    format=FORMAT,
)

train_checkpoint = dict()

if __name__ == "__main__":
    train_path = f"./data/{args['dataset']}/{setting}/train/{args['nbest']}_align_token.json"
    dev_path = f"./data/{args['dataset']}/{setting}/valid/{args['nbest']}_align_token.json"
    # test_path = f"./data/aishell/{setting}/test/{args['nbest']}_align_token.json"
    
    print(f"Prepare data")
    with open(train_path) as f,\
         open(dev_path) as d:
        #  open(test_path) as t:
        train_json = json.load(f)
        dev_json = json.load(d)

    train_set = get_dataset(train_json, tokenizer = tokenizer, data_type = 'align', topk = int(args['nbest']))
    dev_set = get_dataset(dev_json, tokenizer = tokenizer, data_type = 'align', topk = int(args['nbest']))

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

    model = nBestTransformer(
        nBest=args['nbest'],
        device=device,
        lr=float(train_args["lr"]),
        align_embedding=train_args["align_embedding"],
        dataset = args['dataset'],
        from_pretrain = train_args['from_pretrain']
    )

    if (train_args['from_pretrain']):
        pretrain_name = "Pretrain"
    else:
        pretrain_name = "Scratch"

    scheduler = OneCycleLR(
        model.optimizer,
        epochs = int(train_args['epoch']),
        steps_per_epoch = len(train_loader),
        pct_start = 3 / int(train_args['epoch']),
        anneal_strategy='linear',
        max_lr = float(train_args['lr'])
    )
    print(f"training")

    min_val = 1e8

    for e in range( train_args['epoch']):
        model.train()

        logging_loss = 0.0
        model.optimizer.zero_grad()
        for n, data in enumerate(tqdm(train_loader, ncols = 100)):
            # logging.warning(f'token.shape:{token.shape}')
            token = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            label = data['labels'].to(device)

            loss = model(token, mask, label)

            loss /= train_args["accumgrad"]
            loss.backward()
            logging_loss += loss.clone().detach().cpu()

            if ((n + 1) % train_args["accumgrad"] == 0) or ((n + 1) == len(train_loader)):
                model.optimizer.step()
                scheduler.step()                
                model.optimizer.zero_grad()

            if (n + 1) % train_args["print_loss"] == 0 or (n + 1) == len(train_loader):
                logging.warning(
                    f"Training epoch :{e + 1} step:{n + 1}, training loss:{logging_loss}"
                )

                logging_loss = 0.0
        

        train_checkpoint["epoch"] = e + 1
        train_checkpoint["model"] = model.model.state_dict()
        train_checkpoint["embedding"] = model.embedding.state_dict()
        train_checkpoint["linear"] = model.embeddingLinear.state_dict()

        train_checkpoint["optimizer"] = model.optimizer.state_dict()
        train_checkpoint["scheduler"] = scheduler.state_dict()
        
        if not os.path.exists( f"./checkpoint/{args['dataset']}/{args['nbest']}_align/{setting}"):
            os.makedirs(f"./checkpoint/{args['dataset']}/{args['nbest']}_align/{setting}")
        torch.save(
            train_checkpoint,
            f"./checkpoint/{args['dataset']}/{args['nbest']}_align/{setting}/{pretrain_name}_checkpoint_train_{e + 1}.pt",
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

            if val_loss < min_val:
                min_val = val_loss
                torch.save(
                    train_checkpoint,
                    f"./checkpoint/nBestTransformer/{training_mode}/{model_name}/checkpoint_train_best.pt",
                )
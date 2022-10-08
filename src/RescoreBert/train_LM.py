import os
import sys
import json
import torch
from pathlib import Path
from utils.Datasets import get_Dataset
from utils.PrepareModel import prepare_GPT2, prepare_MLM
from utils.LoadConfig import load_config
from transformers import (
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    Trainer
)

config = f'./config/mlm.yaml'

args, train_args, recog_args = load_config(config)
setting = "withLM" if args['withLM'] else "noLM"
lm_name = "MLM" if (args['MLM']) else "CLM"
print(f'LM:{lm_name}')


checkpoint_name = f"./checkpoint/{args['dataset']}/{lm_name}/{setting}/{args['nbest']}"
output_dir = Path(checkpoint_name)
output_dir.mkdir(parents = True, exist_ok = True)
log_name = f"./log/{args['dataset']}/{lm_name}/{setting}/{args['nbest']}"
output_dir = Path(log_name)
output_dir.mkdir(parents = True, exist_ok = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if (args['MLM']):
    model, tokenizer = prepare_MLM(args['dataset'], device)
else:
    model, tokenizer = prepare_GPT2(args['dataset'], device)

if (tokenizer.pad_token is None):
        tokenizer.pad_token = tokenizer.eos_token

print(f'pad_id:{tokenizer.pad_token_id}')

print(f'load data.json')
with open(f"../../data/{args['dataset']}/data/{setting}/train/data.json", "r") as train, \
     open(f"../../data/{args['dataset']}/data/{setting}/dev/data.json", "r") as valid:
     train_json = json.load(train)
     valid_json = json.load(valid)

print(f'prepare dataset')
train_dataset = get_Dataset(train_json, tokenizer, dataset = args['dataset'])
valid_dataset = get_Dataset(valid_json, tokenizer, dataset = args['dataset'])

training_args = TrainingArguments(
    output_dir = checkpoint_name,
    evaluation_strategy = "epoch",
    learning_rate = float(train_args["lr"]),
    per_device_train_batch_size= train_args['train_batch'],
    per_device_eval_batch_size = train_args['valid_batch'],
    num_train_epochs = train_args['epoch'],
    weight_decay = 0.1,
    warmup_steps= 0.1,

    save_strategy = 'epoch',
    logging_dir = log_name, 
    logging_strategy="epoch",

    no_cuda = False,
    dataloader_num_workers = 1,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = args["MLM"])

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset= train_dataset,
    eval_dataset = valid_dataset,
    data_collator = data_collator
)

trainer.train()
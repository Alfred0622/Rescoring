import torch
import glob
import logging
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from src_utils.LoadConfig import load_config
from utils.PrepareModel import prepare_model
from utils.CollateFunc import trainBatch
from utils.Datasets import get_dataset
from src_utils.LoadConfig import load_config

from transformers import Trainer, TrainingArguments, DataCollator

config = f'./config/classification.yaml'

args, train_args, recog_args = load_config(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

setting = 'withLM' if (args['withLM']) else 'noLM'

model, tokenizer = prepare_model(args, train_args, device)

with open(f"../../data/{args['dataset']}/data/{setting}/train/data.json") as f, \
     open(f"../../data/{args['dataset']}/data/{setting}/dev/data.json") as v:
    train_json = json.load(f)
    valid_json = json.load(v)

train_dataset = get_dataset(train_json, tokenizer, nbest = args['nbest'])
valid_dataset = get_dataset(valid_json, tokenizer, nbest = args['nbest'])

training_args = TrainingArguments(
    output_dir = f"./checkpoint/{args['dataset']}/result/{setting}",
    overwrite_output_dir = True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=train_args['train_batch'],
    per_device_eval_batch_size=train_args['valid_batch'],
    gradient_accumulation_steps=train_args['accumgrad'],
    eval_accumulation_steps=1,
    learning_rate=float(train_args['lr']),
    weight_decay=0.01,
    num_train_epochs=train_args['epoch'],
    lr_scheduler_type="linear",
    warmup_ratio=0.1,

    logging_dir=f"./log/{args['dataset']}/{args['nbest']}_{setting}",
    logging_strategy="steps",
    logging_steps = 100,
    logging_first_step=True,
    logging_nan_inf_filter=False,
            
    save_strategy='epoch',
    no_cuda=False,
    dataloader_num_workers=1,
    load_best_model_at_end=True,
    # metric_for_best_model="CE",
    # greater_is_better=False,
    # predict_with_generate=True    
)

data_collator = DataCollator(tokenizer)

trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        tokenizer = tokenizer,
)

trainer.train()
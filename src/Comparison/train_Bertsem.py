import json
import yaml
import random
import torch
import glob
import logging
import os
from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from model.BertForComparison import Bert_Sem
from utils.Datasets import(
    get_dataset
)
from utils.CollateFunc import(
    bertCompareBatch,
    bertCompareRecogBatch,
)
from utils.LoadConfig import load_config
from utils.PrepareModel import prepare_model

from transformers import Trainer, TrainingArguments, DataCollator

args, train_args, recog_args = load_config("./config/comparison.yaml")

dataset = args['dataset']
model_name = args['model_name']

print(f'dataset:{dataset}')
print(f'model_name:{model_name}')

setting = 'withLM' if args['withLM'] else 'noLM'

model, tokenizer = prepare_model(dataset, model_name)

with open(f"./data/{dataset}/train/{setting}/{args['nbest']}best/data.json") as train , \
     open(f"./data/{dataset}/valid/{setting}/{args['nbest']}best/data.json") as valid:
    
     train_json = json.load(train)
     valid_json = json.load(valid)

train_dataset = get_dataset(train_json)
valid_dataset = get_dataset(valid_json)


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

    logging_dir=f"./log/{args['dataset']}/{train_args['nbest']}_{train_args['mode']}",
    logging_strategy="steps",
    logging_steps = 100,
    logging_first_step=True,
    logging_nan_inf_filter=False,
            
    save_strategy='epoch',
    no_cuda=False,
    dataloader_num_workers=1,
    load_best_model_at_end=True,
    metric_for_best_model="BCE",
    greater_is_better=False,
    predict_with_generate=True    
)

data_collator = DataCollator(tokenizer, model = model)

trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        tokenizer = tokenizer,
)

trainer.train()
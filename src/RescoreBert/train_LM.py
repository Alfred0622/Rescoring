import os
import sys
sys.path.append("../")
import json
import torch
import random
from pathlib import Path
from utils.Datasets import get_Dataset
from utils.PrepareModel import prepare_GPT2, prepare_MLM
from src_utils.LoadConfig import load_config
from transformers import (
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    Trainer
)



assert (len(sys.argv) == 2), f'Usage: python ./train_LM.py <CLM or MLM>'

lm_name = sys.argv[1].strip().upper()

if (lm_name == 'CLM'):
    config = f'./config/clm.yaml'
elif (lm_name == 'MLM'):
    config =  f'./config/mlm.yaml'
else:
    assert (lm_name in ["CLM", "MLM"]), f'Usage: python ./train_LM.py <CLM or MLM>'

args, train_args, recog_args = load_config(config)
setting = "withLM" if args['withLM'] else "noLM"
lm_name = "MLM" if (lm_name == 'MLM') else "CLM"
print(f'LM:{lm_name}')

if (args['dataset'] in ['csj']):
    if (args['jp_split']):
        lm_name = f'{lm_name}_char'
    else:
        lm_name = f"{lm_name}"
else:
    lm_name = f"{lm_name}"

checkpoint_name = f"./checkpoint/{args['dataset']}/{lm_name}"
output_dir = Path(checkpoint_name)
output_dir.mkdir(parents = True, exist_ok = True)
log_name = f"./log/{args['dataset']}/{lm_name}/{setting}"
output_dir = Path(log_name)
output_dir.mkdir(parents = True, exist_ok = True)

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if (args['MLM']):
    model, tokenizer = prepare_MLM(args['dataset'], device)
else:
    model, tokenizer = prepare_GPT2(args['dataset'], device)

if (tokenizer.pad_token is None):
        tokenizer.pad_token_id = 0

print(f'pad_id:{tokenizer.pad_token_id}')

if (args['dataset'] in ['aishell2']):
    valid_path = 'dev_ios'
elif (args['dataset'] in ['tedlium2']):
    valid_path = 'dev_trim'
elif (args['dataset'] in ['librispeech']):
    valid_path = 'valid'
else:
    valid_path = 'dev'

print(f'load data.json')
with open(f"../../data/{args['dataset']}/data/{setting}/train/data.json", "r") as train, \
     open(f"../../data/{args['dataset']}/data/{setting}/{valid_path}/data.json", "r") as valid:
     train_json = json.load(train)
     valid_json = json.load(valid)
    
print(type(train_json))
print(len(train_json))

print(f'prepare dataset')
train_dataset = get_Dataset(train_json, tokenizer, lm = lm_name, dataset = args['dataset'], jp_split=args['jp_split'])
valid_dataset = get_Dataset(valid_json, tokenizer, lm = lm_name, dataset = args['dataset'], jp_split=args['jp_split'])

# exit()

print(len(train_dataset))
print(len(valid_dataset))

training_args = TrainingArguments(
    output_dir = checkpoint_name,
    overwrite_output_dir= True,
    evaluation_strategy = "epoch",
    learning_rate = float(train_args["lr"]),
    per_device_train_batch_size= train_args['train_batch'],
    per_device_eval_batch_size = train_args['valid_batch'],
    num_train_epochs = train_args['epoch'],
    warmup_ratio = 0.1 ,
    lr_scheduler_type = 'linear',
    seed = 42,

    logging_dir = log_name, 
    logging_strategy="steps",
    logging_steps = 500,
    logging_first_step=True,
    logging_nan_inf_filter=True,
    group_by_length = True,
    
    save_strategy = 'epoch',
    no_cuda = False,
    dataloader_num_workers = 1,
    greater_is_better=False,
    gradient_accumulation_steps = int(train_args['accumgrad'])
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = args["MLM"])

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset= train_dataset,
    eval_dataset = valid_dataset,
    data_collator = data_collator,
    tokenizer = tokenizer
)

trainer.train()
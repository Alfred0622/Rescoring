import os
import sys
sys.path.append("..")
sys.path.append("../..")
import json
import logging
import numpy as np
from transformers import (
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.Datasets import get_dataset
from utils.CollateFunc import trainBatch
from src_utils.LoadConfig import load_config
from utils.PrepareModel import prepare_model
from jiwer import wer
import gc

tqdm = partial(tqdm, ncols=100)

task_name = sys.argv[1]

if (task_name == 'align_concat'):
    config_name = './config/nBestAlign.yaml'
    topk = 1
elif (task_name == 'plain'):
    config_name = './config/nBestPlain.yaml'
    topk = -1
else:
    config_name = './config/Bart.yaml'
    topk = -1

def compute_metric(eval_pred):
    hyp, ref = eval_pred

    decoded_preds = tokenizer.batch_decode(hyp, skip_special_tokens=True)
    
    if (args['dataset'] in ['csj']):
        for i, pred in enumerate(decoded_preds):
            new_pred = "".join(pred.split())
            new_pred = [p for p in new_pred]
            # assert(len(new_pred) > 0) , f"empty hyp at {i} = {pred}"
            decoded_preds[i] = " ".join(new_pred)

    # Replace -100 in the labels as we can't decode them.
    # labels[ref == -100] = tokenizer.pad_token_id
    labels = np.where(ref != -100, ref, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    if (args['dataset'] in ['csj']): 
        for i, ref in enumerate(decoded_labels):
            new_ref = "".join(ref.split())
            new_ref = [r for r in new_ref]

            decoded_labels[i] = " ".join(new_ref)

    print(f"hyp:{len(decoded_preds)}")
    print(f"ref:{len(decoded_labels)}")

    print(f"decoded_preds:{type(decoded_preds)} -- {decoded_preds[-3:]}")
    print(f"labels:{type(decoded_labels)} -- {decoded_labels[-3:]}")

    print(f'wer:{wer(decoded_labels, decoded_preds)}')

    return {"wer": wer(decoded_labels, decoded_preds)}

args, train_args, recog_args = load_config(config_name)
print(f"from_pretrain:{train_args['from_pretrain']}")

model, tokenizer = prepare_model(args['dataset'], from_pretrain = train_args['from_pretrain'])
setting = 'withLM' if args['withLM'] else 'noLM'

if (args['dataset'] == 'old_aishell'):
    setting = ""

sep_token = train_args['sep_token'] if train_args['sep_token'] is not None else tokenizer.eos_token
print(f'sep_token:{sep_token}')

if (args['dataset'] in ['aishell2']):
    dev_set = 'dev_ios'
elif (args['dataset'] in ['librispeech']):
    dev_set = 'valid'
else:
    dev_set = 'dev'

if (train_args['from_pretrain']):
    pretrain_name = ""
else:
    pretrain_name = "noPretrain"

if (task_name == 'plain'):
    if (sep_token == '[SEP]'):
        sep_token_name = 'sep'
    else:
        sep_token_name = sep_token
    
    task_name = f'{task_name}_{sep_token_name}_{pretrain_name}'

else:
    task_name = f'{task_name}_{pretrain_name}'

if (not os.path.exists(f"./checkpoint/{args['dataset']}/{args['nbest']}_{task_name}/{setting}")):
    os.makedirs(f"./checkpoint/{args['dataset']}/{args['nbest']}_{task_name}/{setting}")
if (not os.path.exists(f"./log/{args['dataset']}/{args['nbest']}_{task_name}/{setting}")):
    os.makedirs(f"./log/{args['dataset']}/{args['nbest']}_{task_name}/{setting}")

train_path = f"../../data/{args['dataset']}/data/{setting}/train/data.json"
valid_path = f"../../data/{args['dataset']}/data/{setting}/{dev_set}/data.json"

with open(train_path) as train, open(valid_path) as valid:
    train_json = json.load(train)
    valid_json = json.load(valid)

if (topk < 0):
    topk = args['nbest']

if (train_args['data_type'] == 'single'):
    valid_topk = 1
else:
    valid_topk = topk

print(f'prepare data & tokenization')
valid_dataset = get_dataset(valid_json, args['dataset'], tokenizer, data_type = train_args['data_type'], sep_token = sep_token, topk = valid_topk, for_train = True)
train_dataset = get_dataset(train_json, args['dataset'], tokenizer, data_type = train_args['data_type'], sep_token = sep_token, topk = topk, for_train = True)


del train_json
del valid_json
gc.collect()

training_args = Seq2SeqTrainingArguments(
            output_dir=f"./checkpoint/{args['dataset']}/{args['nbest']}_{task_name}/{setting}/result",
            overwrite_output_dir=True,
            evaluation_strategy='epoch',
            prediction_loss_only=False,
            per_device_train_batch_size=train_args['train_batch'],
            per_device_eval_batch_size=train_args['valid_batch'],
            gradient_accumulation_steps=train_args['accumgrad'],
            # eval_accumulation_steps=1,
            learning_rate=float(train_args['lr']),
            weight_decay=0.02,
            num_train_epochs=train_args['epoch'],
            lr_scheduler_type="linear",
            warmup_ratio = 0.02,

            logging_dir=f"./log/{args['dataset']}/{args['nbest']}_{task_name}/{setting}",
            logging_strategy="steps",
            logging_steps = 2000,
            logging_first_step=True,
            logging_nan_inf_filter=False,
            
            save_strategy='epoch',
            no_cuda=False,
            dataloader_num_workers=4,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            predict_with_generate=True,
            generation_num_beams = 5,
            generation_max_length = 150,
            run_name = f"{args['dataset']}/{args['nbest']}_{task_name}/{setting}"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        tokenizer = tokenizer,
        compute_metrics= compute_metric
)

trainer.train()

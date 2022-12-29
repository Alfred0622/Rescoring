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
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.Datasets import get_dataset
from utils.CollateFunc import trainBatch
from utils.LoadConfig import load_config
from utils.PrepareModel import prepare_model
from jiwer import wer

task_name = sys.argv[1]

if (task_name == 'align_concat'):
    config_name = './config/nBestAlign.yaml'
    topk = 1
elif (task_name == 'plain'):
    config_name = './config/nBestPlain.yaml'
    topk = 1
else:
    config_name = './config/Bart.yaml'
    topk = -1

def compute_metric(eval_pred):
    hyp, ref = eval_pred
    # logging.warning(hyp.shape, type(hyp))
    decoded_preds = tokenizer.batch_decode(hyp, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    # labels[ref == -100] = tokenizer.pad_token_id
    labels = np.where(ref != -100, ref, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return {"wer": wer(decoded_labels, decoded_preds)}

args, train_args, recog_args = load_config(config_name)
model, tokenizer = prepare_model(args['dataset'])
setting = 'withLM' if args['withLM'] else 'noLM'

if (args['dataset'] == 'old_aishell'):
    setting = ""

if (args['dataset'] in ['aishell2']):
    dev_set = 'dev_ios'
elif (args['dataset'] in ['librispeech']):
    dev_set = 'dev_clean'
else:
    dev_set = 'dev'

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
train_dataset = get_dataset(train_json, tokenizer, data_type = train_args['data_type'] ,topk = topk, for_train = True)
valid_dataset = get_dataset(valid_json, tokenizer, data_type = train_args['data_type'], topk = valid_topk, for_train = True)

training_args = Seq2SeqTrainingArguments(
            output_dir=f"./checkpoint/{args['dataset']}/{args['nbest']}_{task_name}/{setting}/result",
            overwrite_output_dir=True,
            evaluation_strategy='epoch',
            prediction_loss_only=False,
            per_device_train_batch_size=train_args['train_batch'],
            per_device_eval_batch_size=train_args['valid_batch'],
            gradient_accumulation_steps=train_args['accumgrad'],
            eval_accumulation_steps=1,
            learning_rate=float(train_args['lr']),
            weight_decay=0.1,
            num_train_epochs=train_args['epoch'],
            lr_scheduler_type="linear",
            warmup_ratio = 0.3,

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
            predict_with_generate=True
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

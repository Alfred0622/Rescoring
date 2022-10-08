import sys
sys.path.append("..")
import torch
import logging
from transformers import (
    AutoModelForSeq2SeqLM,
    BertTokenizer,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
from model.BertForComparison import BertForComparison

def prepare_model(args, train_args, device):
    if (args["dataset"] in {'aishell', 'aishell2'}):
        pretrain_name = 'bert-base-chinese'
    elif (args["dataset"] in {'tedlium2', 'librispeech'}):
        pretrain_name = 'bert-base-uncased'
    elif (args["dataset"] in {'csj'}):
        pass

    if (args["model_name"] == 'sem'):
        model = BertForComparison(args["dataset"], device, float(train_args["lr"]))
    elif (args["model_name"] == 'alsem'):
        model = None
    
    tokenizer = BertTokenizer.from_pretrained(pretrain_name)

    return model, tokenizer

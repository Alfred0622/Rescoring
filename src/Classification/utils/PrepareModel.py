import sys
sys.path.append("..")
import torch
import logging
from transformers import (
    BertForSequenceClassification,
    BertTokenizer
)

def prepare_model(args, train_args, device):
    if (args["dataset"] in {'aishell', 'aishell2'}):
        pretrain_name = 'bert-base-chinese'
    elif (args["dataset"] in {'tedlium2', 'librispeech'}):
        pretrain_name = 'bert-base-uncased'
    elif (args["dataset"] in {'csj'}):
        pass

    model = BertForSequenceClassification.from_pretrained(pretrain_name, num_labels = args['nbest'])
    
    tokenizer = BertTokenizer.from_pretrained(pretrain_name)

    return model, tokenizer

import sys
sys.path.append("..")
import torch
import logging
from transformers import (
    BertForSequenceClassification,
    AutoModelForSeq2SeqLM,
    BertTokenizer,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
from model.BertForComparison import Bert_Sem, Bert_Alsem

def prepare_model(args, train_args, device):
    if (args["dataset"] in {'aishell', 'aishell2'}):
        pretrain_name = 'bert-base-chinese'
    elif (args["dataset"] in {'tedlium2', 'librispeech', 'tedlium2_conformer'}):
        pretrain_name = 'bert-base-uncased'
    elif (args["dataset"] in {'csj'}):
        pass

    if (args["model_name"] == 'sem'):
        model = Bert_Sem(args['dataset'], device, lr = float(train_args['lr']))
    elif (args["model_name"] == 'alsem'):
        model = Bert_Alsem(pretrain_name, device,ctc_weight = train_args['ctc_weight'])
    
    tokenizer = BertTokenizer.from_pretrained(pretrain_name)

    return model, tokenizer

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
    DataCollatorForSeq2Seq,
    BertJapaneseTokenizer
)
from model.BertForComparison import Bert_Sem, Bert_Alsem

def prepare_model(args, train_args, device):
    print(f"{args['model_name']}")
    if (args["dataset"] in {'aishell', 'aishell2'}):
        pretrain_name = 'bert-base-chinese'
    elif (args["dataset"] in {'tedlium2', 'librispeech', 'tedlium2_conformer'}):
        pretrain_name = 'bert-base-uncased'
    elif (args["dataset"] in {'csj'}):
        pretrain_name = 'cl-tohoku/bert-base-japanese'

    if (args["model_name"] == 'sem'):
        print(f"sem")
        model = Bert_Sem(args['dataset'], device)
    elif (args["model_name"] == 'alsem'):
        model = Bert_Alsem(args['dataset'],device,hidden_size = 1024,ctc_weight = train_args['ctc_weight'][args['dataset']])
    

    if (args["dataset"] in {'csj'}):
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    else:
        tokenizer = BertTokenizer.from_pretrained(pretrain_name)

    return model, tokenizer

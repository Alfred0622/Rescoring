import torch
import logging
from transformers import (
    AutoModelForSeq2SeqLM,
    BertTokenizer,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)

def prepare_model(dataset):
    if (dataset in ['aishell', 'aishell2', 'old_aishell']):
        model = AutoModelForSeq2SeqLM.from_pretrained(f'fnlp/bart-base-chinese')
        tokenizer = BertTokenizer.from_pretrained(f'fnlp/bart-base-chinese')
    elif (dataset in ['tedlium2', 'librispeech']): # english
        pass

    elif (dataset in ['csj']): # japanese
        pass

    return model, tokenizer
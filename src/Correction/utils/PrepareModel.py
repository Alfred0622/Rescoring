import sys
sys.path.append("..")
# from models import nBestAlignBart
import torch
import logging
from transformers import (
    AutoModelForSeq2SeqLM,
    BertTokenizer,
    BartTokenizer,
)
from transformers import Trainer

class nBestAlignTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        loss = model(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).loss

        return loss

def prepare_model(dataset):
    print(f'dataset:{dataset}')
    if (dataset in ['aishell', 'aishell2', 'old_aishell']):
        print(f'bart-base-chinese')
        model = AutoModelForSeq2SeqLM.from_pretrained(f'fnlp/bart-base-chinese')
        tokenizer = BertTokenizer.from_pretrained(f'fnlp/bart-base-chinese')
    elif (dataset in ['tedlium2', 'librispeech']): # english
        model = AutoModelForSeq2SeqLM.from_pretrained(f'facebook/bart-base')
        tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-base')

    elif (dataset in ['csj']): # japanese
        pass

    return model, tokenizer
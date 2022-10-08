import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM, 
    BertTokenizerFast,
    AutoTokenizer,
    GPT2Tokenizer
)

def prepare_GPT2(dataset, device):
    if (dataset in ['aishell', 'aishell2']):
        model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese').to(device)
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    if (dataset in ['tedlium2', 'librispeech']):
        model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if (dataset in ['csj']):
        pass
    
    return model, tokenizer

def prepare_MLM(dataset, device):
    if (dataset in ['aishell', 'aishell2']):
        model = AutoModelForMaskedLM.from_pretrained('bert-base-chinese').to(device)
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    if (dataset in ['tedlium2', 'librispeech']):
        model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased').to(device)
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    if (dataset in ['csj']):
        pass

    return model, tokenizer
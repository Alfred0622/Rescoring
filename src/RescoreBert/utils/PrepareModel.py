import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BertModel,
    BertForMaskedLM ,
    BertTokenizer,
    BertTokenizerFast,
    AutoTokenizer,
    GPT2Tokenizer
)
from torch.optim import AdamW

class RescoreBert(torch.nn.Module):
    def __init__(self, pretrain_name,device):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).to(device)
        self.linear = torch.nn.Linear(768, 1)

        self.l2_loss = torch.nn.MSELoss()
    
    def forward(self, input_ids, attention_mask, labels = None):
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).pooler_output

        # print(f'output:{output.shape}')

        score = self.linear(output)
        
        score = score.squeeze(-1)
        if (labels is not None):
            labels = labels.to(dtype = torch.float32)

            # print(f'labels:{labels}')

            # print(f'score in forward:{score.dtype}')
            # print(f'labels in forward:{labels.dtype}')

            loss = self.l2_loss(
                score, labels
            )
        else:
            loss = None

        # loss = loss.to(dtype = torch.float32)

        return {"score": score, "loss": loss}

    def parameters(self):
        return list(self.bert.parameters()) + list(self.linear.parameters())

def prepare_GPT2(dataset, device):
    if (dataset in ['aishell', 'aishell2']):
        model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese').to(device)
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    if (dataset in ['tedlium2', 'librispeech']):
        model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if (dataset in ['csj']):
        model = AutoModelForCausalLM.from_pretrained('ClassCat/gpt2-base-japanese-v2').to(device)
        tokenizer = AutoTokenizer.from_pretrained('ClassCat/gpt2-base-japanese-v2')
    
    return model, tokenizer

def prepare_MLM(dataset, device):
    if (dataset in ['aishell', 'aishell2']):
        model = BertForMaskedLM.from_pretrained('bert-base-chinese').to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    if (dataset in ['tedlium2', 'librispeech']):
        model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if (dataset in ['csj']):
        model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese').to(device)
        tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

    return model, tokenizer

def prepare_RescoreBert(dataset, device):
    if (dataset in ['aishell', 'aishell2']):
        model = RescoreBert('bert-base-chinese', device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    elif (dataset in ['tedlium2', 'librispeech']):
        model = RescoreBert('bert-base-uncased', device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return model, tokenizer
    
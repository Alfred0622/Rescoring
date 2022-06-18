import torch
import torch.nn as nn
from torch.optim import AdamW
import logging
from transformers import BertTokenizer,BartForConditionalGeneration

class AudioAwareBart(nn.Module):
    def __init__(self):
        super().__init__(self, device)

        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
        self.model = BartForConditionalGeneration.from_pretrained(
            "fnlp/bart-base-chinese"
        ).to(self.device)
    
    def forward():
        pass
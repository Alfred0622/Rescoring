import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import logging
from transformers import BertTokenizer, BartForConditionalGeneration, BartConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AdamWeightDecay

class RoBart(nn.Module):
    def __init__(self, device, lr=1e-5):
        nn.Module.__init__(self)
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
        self.model = BartForConditionalGeneration.from_pretrained(
            "fnlp/bart-base-chinese"
        ).to(self.device)

        # self.model = AutoModelForSeq2SeqLM.from_pretrained('fnlp/bart-base-chinese').to(self.device)
        # self.tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')

        logging.warning(self.model.config)

        self.optimizer = AdamW(self.model.parameters(), lr = lr, eps = 1e-6, weight_decay = 0.1)

        logging.warning(self.model)

    def forward(self, input_id, attention_masks, labels, segments = None):
        output = self.model(
            input_ids = input_id,
            attention_mask = attention_masks, 
            labels=labels,
            return_dict = True
        )

        loss = output.loss

        return loss

    def recognize(self, input_id, attention_masks, segments = None ,max_lens=50):
        
        output = self.model.generate(
            input_ids = input_id,
            attention_mask = attention_masks,
            top_k = 5,
            max_length=max_lens,
        )

        return output

import torch
import logging
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertConfig,
)
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer

class BertForComparison(torch.nn.Module):
    def __init__(self):
        super().__init__(self, lr)
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.lr = lr
        self.optimizer = AdamW(lr = self.lr)
        self.linear = torch.nn.Linear(768, 2)
        self.loss = CrossEntropyLoss()

    def forward(self, input_ids, segment, attention_mask, labels):
        total_loss = 0.0

        cls = self.model(
            input_ids=input_id, attention_mask=attention_mask
        )

        score = self.linear(cls)
        score = softmax(score)

        loss = CrossEntropyLoss(score, labels)

        return loss
    
    def recognize(self, input_ids, segment, attention_mask):
        cls = self.model(
            input_ids=input_id, attention_mask=attention_mask
        )

        score = self.linear(cls)
        score = softmax(score)

        return score
import torch
import logging
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss, BCELoss, Sigmoid
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertConfig,
)
from torch.optim import AdamW

class BertForComparison(torch.nn.Module): # Bert_sem
    def __init__(self, lr = 1e-5):
        torch.nn.Module.__init__(self)
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.lr = lr
        self.optimizer = AdamW(self.model.parameters(), lr = self.lr)
        self.linear = torch.nn.Linear(768, 1)
        self.loss = BCELoss()
        self.sigmoid = Sigmoid()

    def forward(self, input_ids, segment, attention_mask, labels):
        total_loss = 0.0

        cls = self.model(
            input_ids=input_ids, 
            token_type_ids = segment,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        score = self.linear(cls)
        # logging.warning(f'score:{score.shape}')
        score = self.sigmoid(score).squeeze(-1)
        
        loss = self.loss(score, labels)

        return loss
    
    def recognize(self, input_ids, segment, attention_mask):
        cls = self.model(
            input_ids=input_ids, 
            token_type_ids = segment,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        score = self.linear(cls)
        score = self.sigmoid(score)

        return score

# class BertForComparason_AL(torch.nn.Module): # Bert Alsem
#     def __init__(self, pretrain_name ,lr = 1e-5):
#         torch.nn.Module.__init__(self)
#         self.bert = BertModel.from_pretrained(pretrain_name)
#         self.rnn = 
#         self.lr = lr


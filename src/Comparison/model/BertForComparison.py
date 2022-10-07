import torch
import logging
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss, BCELoss, Sigmoid
from torch.nn import LSTM, Linear
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertConfig,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.optim import Adam, AdamW

class BertForComparison(torch.nn.Module): # Bert_sem
    def __init__(self, dataset, device, lr = 1e-5):
        torch.nn.Module.__init__(self)
        self.dataset = dataset
        if (self.dataset in ['aishell', 'aishell2']):
            self.model = BertModel.from_pretrained('bert-base-chinese')
        elif (self.dataset in ['tedlium2', 'librispeech']):
            self.model = BertModel.from_pretrained('bert-base-uncased')
        elif (self.dataset in ['csj']):
            pass
            # self.model = BertModel.from_pretrained()
        self.model = self.model.to(device)
        
        self.linear = torch.nn.Linear(768, 1).to(device)
        self.loss = BCELoss()
        self.sigmoid = Sigmoid()

        parameters = list(self.model.parameters()) + list(self.linear.parameters())

        self.optimizer = AdamW(parameters, lr = lr)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        total_loss = 0.0

        cls = self.model(
            input_ids=input_ids, 
            token_type_ids = token_type_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        score = self.linear(cls)
        # logging.warning(f'score:{score.shape}')
        score = self.sigmoid(score).squeeze(-1)

        if (labels is not None):
            loss = self.loss(score, labels)
        else:
            loss = None

        return SequenceClassifierOutput(
            loss = loss,
            logits = score
        )
    
    def recognize(self, input_ids, token_type_ids, attention_mask):
        cls = self.model(
            input_ids=input_ids, 
            token_type_ids = token_type_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        score = self.linear(cls)
        score = self.sigmoid(score)
        return score

class BertForComparason_AL(torch.nn.Module): # Bert Alsem
    def __init__(
        self, 
        pretrain_name, 
        hidden_size = 2048, 
        output_size = 64,
        dropout = 0.3,
        lr = 1e-5
    ):
        torch.nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_name)
        self.rnn = torch.nn.LSTM(
            input_size = 768,
            hidden_size = hidden_size,
            num_layers = 1,
            batch_first = True,
            dropout = dropout,
            bidirectional = True,
            proj_size = output_size
        ) # output: 64 * 2(bidirectional)
        
        # input: 128(output) + 4 (am_score + lm_score of the two hyps)
        self.fc1 = torch.nn.Sequntial(
            Linear(132, 64),
            Linear(64, 1),
            torch.nn.ReLU()
        )

        self.fc2 = torch.nn.Linear(
            Linear(),
            Linear(),
            Sigmoid()
        )

        self.lr = lr
        
        parameters = list(self.bert.parameters()) + \
                     list(self.rnn.parameters()) + \
                     list(self.fc1.parameters()) + \
                     list(self.fc2.parameters())

        self.optimizer = Adam(parameters, lr = lr)

        return

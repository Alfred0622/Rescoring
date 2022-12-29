import torch
import logging
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss, BCELoss, Sigmoid
from torch.nn import LSTM, Linear
from torch.nn import AvgPool1d, MaxPool1d
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    DistilBertModel,
    DistilBertConfig,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.optim import Adam, AdamW


class Bert_Compare(torch.nn.Module): # Bert_sem
    def __init__(self, dataset, device, lr = 1e-5):
        torch.nn.Module.__init__(self)
        self.dataset = dataset
        self.config = BertConfig(num_labels = 1)
        if (self.dataset in ['aishell', 'aishell2']):
            self.model = BertForSequenceClassification(self.config).from_pretrained('bert-base-chinese')
        elif (self.dataset in ['tedlium2', 'librispeech']):
            self.model = BertForSequenceClassification(self.config).from_pretrained('bert-base-uncased')
        elif (self.dataset in ['csj']):
            pass
            # self.model = BertModel.from_pretrained()
        self.model = self.model.to(device)

        self.loss = BCELoss()
        self.sigmoid = Sigmoid()

        parameters = list(self.model.parameters())

        self.optimizer = AdamW(parameters, lr = lr)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        loss = self.model(
            input_ids=input_ids, 
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
            labels = labels,
            return_dict = True
        ).loss

        return SequenceClassifierOutput(
            loss = loss,
            logits = None
        )
    
    def recognize(self, input_ids, token_type_ids, attention_mask):
        score = self.model(
            input_ids=input_ids, 
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
            return_dict = True
        ).logits

        score = score.squeeze(-1)# [B, 1] -> [B]
        score = self.sigmoid(score)
        return score

class Bert_Sem(torch.nn.Module): # Bert_sem
    def __init__(self, dataset, device, lr = 1e-5):
        torch.nn.Module.__init__(self)
        self.dataset = dataset
        
        if (self.dataset in ['aishell', 'aishell2']):
            self.config = BertConfig.from_pretrained('bert-base-chinese')
            self.config.hidden_dropout_prob = 0.3
            self.config.attention_probs_dropout_prob = 0.3

            self.bert = BertModel.from_pretrained('bert-base-chinese', config = self.config)
        elif (self.dataset in ['tedlium2', 'librispeech', 'tedlium2_conformer']):
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.config.hidden_dropout_prob = 0.3
            self.config.attention_probs_dropout_prob = 0.3
    
            self.bert = BertModel.from_pretrained('bert-base-uncased', config = self.config)
        elif (self.dataset in ['csj']):
            pass
            # self.model = BertModel.from_pretrained()
        self.bert = self.bert.to(device)
        
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 1).to(device)
        self.loss = BCELoss()
        self.sigmoid = Sigmoid()

        parameters = list(self.bert.parameters()) + list(self.linear.parameters())

        # logging.warning(f'{self.bert}\n {self.linear}')

        self.optimizer = AdamW(parameters, lr = lr)

    def forward(self, input_ids, token_type_ids, attention_mask, labels = None):
        cls = self.bert(
            input_ids=input_ids, 
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
            return_dict = True
        ).pooler_output # [B,768]

        cls = self.dropout(cls)
        score = self.linear(cls).squeeze(1) # [B, 768] -> [B, 1] -> [B]

        score = self.sigmoid(score)

        if (labels is not None):
            loss = self.loss(score, labels)
        else:
            loss = None

        return SequenceClassifierOutput(
            loss = loss,
            logits = score
        )
    
    def recognize(self, input_ids, token_type_ids, attention_mask):
        cls = self.bert(
            input_ids=input_ids, 
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
            return_dict = True
        ).pooler_output# [B, 1, 768] -> [B, 768]

        score = self.linear(cls).squeeze(1)# [B, 768] -> [B, 1] -> [B]
        score = self.sigmoid(score)
        return score

class Bert_Alsem(torch.nn.Module): # Bert Alsem
    def __init__(
        self, 
        pretrain_name, 
        device,
        hidden_size = 2048, 
        output_size = 64,
        ctc_weight = 0.3,
        dropout = 0.3,
        lr = 1e-5
    ):
        torch.nn.Module.__init__(self)
        self.device = device
        self.bert = BertModel.from_pretrained(pretrain_name).to(device)
        self.rnn = torch.nn.LSTM(
            input_size = 768,
            hidden_size = hidden_size,
            num_layers = 1,
            batch_first = True,
            # dropout = dropout,
            bidirectional = True,
            proj_size = output_size
        ).to(device) # output: 64 * 2(bidirectional)
        
        # input: 128(output) + 4 (am_score & lm_score of the two hyps)

        self.fc1 = torch.nn.Sequential(
            Linear(256, 128),
            torch.nn.ReLU()
        ).to(device)

        self.fc2 = torch.nn.Sequential(
            Linear(132, 64),
            Linear(64, 1),
            torch.nn.ReLU()
        ).to(device)

        self.lr = lr
        
        parameters = list(self.bert.parameters()) + \
                     list(self.rnn.parameters()) + \
                     list(self.fc1.parameters()) + \
                     list(self.fc2.parameters())

        self.optimizer = Adam(parameters, lr = lr)

        self.sigmoid = Sigmoid()
        self.loss = BCELoss()

        self.ctc_weight = ctc_weight

    def forward(self, input_ids,token_type_ids ,attention_mask, am_score, ctc_score, lm_score ,labels = None):
        
        last_hidden_states = self.bert(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            return_dict = True
        ).last_hidden_state # [B, L, 768]

        LSTM_state, (h, c) = self.rnn(last_hidden_states) # (B, L, 128)

        # print(f'LSTM_state:{LSTM_state.shape}')

        avg_pool = AvgPool1d(LSTM_state.shape[1]).to(self.device)
        max_pool = MaxPool1d(LSTM_state.shape[1]).to(self.device)

        avg_state = avg_pool(torch.transpose(LSTM_state,1,2))
        max_state = max_pool(torch.transpose(LSTM_state,1,2))

        # print(f'avg_pool:{avg_state.shape}')
        # print(f'max_pool:{max_state.shape}')

        concat_state = torch.cat([torch.transpose(avg_state, 1,2), torch.transpose(max_state,1,2)], dim = -1) # (B, 128 + 128)
        concat_state = concat_state.squeeze(1)
        # print(f'concat_state:{concat_state.shape}')
        concat_state = self.fc1(concat_state) # (B, 256) -> (B, 128)

        am_score = (1 - self.ctc_weight) * am_score + self.ctc_weight * ctc_score

        concat_state = torch.cat([concat_state, am_score], dim = -1) #(B, 128) -> (B, 130)
        concat_state = torch.cat([concat_state, lm_score], dim = -1) #(B, 130) -> (B, 132)

        logits = self.fc2(concat_state).squeeze(-1) #(B, 132) -> (B, 1) ->"squeeze" (B)

        logits = self.sigmoid(logits)

        if (labels is not None):
            loss = self.loss(logits, labels)
        else:
            loss = None
        
        return SequenceClassifierOutput(
            loss = loss,
            logits = logits
        )


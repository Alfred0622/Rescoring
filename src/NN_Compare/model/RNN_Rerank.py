import torch
import logging
import torch.nn as nn
from torch.nn import LSTM, Linear
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import Adam
from transformers.modeling_outputs import SequenceClassifierOutput

class RNN_Reranker(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_layers,
        output_dim,
        device, 
        lr, 
        add_additional_feat = False,
        add_am_lm_score = False,

    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim = 100, padding_idx = 0).to(device)
        if (add_additional_feat):
            input_dim = 117
        elif (add_am_lm_score): # add AM, CTC, LM scores
            input_dim = 103
        else:
            input_dim = 100
        
        self.add_am_lm_score = add_am_lm_score
        self.rnn = LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = False,
            proj_size = output_dim
        ).to(device)

        output_dim = hidden_dim if output_dim == 0 else output_dim

        self.fc = Linear(
            output_dim * 2,
            2
        ).to(device)
        
        embedding_param = sum(p.numel() for p in self.embedding.parameters())
        rnn_param = sum(p.numel() for p in self.rnn.parameters())
        fc_param = sum(p.numel() for p in self.fc.parameters())

        total_param = list(self.embedding.parameters()) + list(self.rnn.parameters()) + list(self.fc.parameters())
        
        logging.warning(f'# of params in Embedding:{embedding_param}')
        logging.warning(f'# of params in RNN:{rnn_param}')
        logging.warning(f'# of params in fc:{fc_param}')
        logging.warning(f'# of params in total:{embedding_param + rnn_param + fc_param}')
        
        self.softmax = nn.Softmax(dim = -1)
        self.optimizer = Adam(total_param, lr = lr)
        self.loss = CrossEntropyLoss()
    
    def forward(self, input_ids_1, input_ids_2,am_score, ctc_score ,lm_score, labels = None):
        input_embeds_1 = self.embedding(input_ids_1)
        input_embeds_2 = self.embedding(input_ids_2)


        if (self.add_am_lm_score):
            am_score_1 = am_score[:, 0].unsqueeze(-1).unsqueeze(-1)
            am_score_2 = am_score[:, 1].unsqueeze(-1).unsqueeze(-1)
            ctc_score_1 = ctc_score[:, 0].unsqueeze(-1).unsqueeze(-1)
            ctc_score_2 = ctc_score[:, 1].unsqueeze(-1).unsqueeze(-1)
            lm_score_1 = lm_score[:, 0].unsqueeze(-1).unsqueeze(-1)
            lm_score_2 = lm_score[:, 1].unsqueeze(-1).unsqueeze(-1)

            am_score_1 = am_score_1.expand(-1, input_embeds_1.shape[1],1)
            ctc_score_1 = ctc_score_1.expand(-1, input_embeds_1.shape[1],1)
            lm_score_1 = lm_score_1.expand(-1, input_embeds_1.shape[1],1)
            am_score_2 = am_score_2.expand(-1, input_embeds_2.shape[1],1)
            ctc_score_2 = ctc_score_2.expand(-1, input_embeds_2.shape[1],1)
            lm_score_2 = lm_score_2.expand(-1, input_embeds_2.shape[1],1)

            input_embeds_1 = torch.cat((input_embeds_1, am_score_1), dim = -1)
            input_embeds_1 = torch.cat((input_embeds_1, ctc_score_1), dim = -1)
            input_embeds_1 = torch.cat((input_embeds_1, lm_score_1), dim = -1)

            input_embeds_2 = torch.cat((input_embeds_2, am_score_2), dim = -1)
            input_embeds_2 = torch.cat((input_embeds_2, am_score_2), dim = -1)
            input_embeds_2 = torch.cat((input_embeds_2, am_score_2), dim = -1)


        output_1, (h1, c1) = self.rnn(input_embeds_1)
        _, (h2, c2) = self.rnn(input_embeds_2)
        
        # print(output_1[:, -1, :].squeeze(1) == h1.squeeze(0))
        # print(f'h_1:{h1.shape}')
        # print(f'h_2:{h2.shape}')

        concat_state = torch.cat((h1,h2), dim = -1)

        output = self.fc(concat_state).squeeze(0)

        logit = self.softmax(output)

        if (labels is not None):
            # print(labels.shape)
            # labels = one_hot(labels, num_classes = 2)
            # print(labels.shape)
            loss = self.loss(logit, labels)
        else:
            loss = None

        return SequenceClassifierOutput(
            loss = loss,
            logits = logit
        )


    
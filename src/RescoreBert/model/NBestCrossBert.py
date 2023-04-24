import sys
sys.path.append("../")
import numpy as np
import torch
import logging
from torch.nn.functional import log_softmax
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertConfig,
    AutoModelForCausalLM
)
from transformers.modeling_outputs import SequenceClassifierOutput
from src_utils.getPretrainName import getBertPretrainName
from torch.nn.functional import log_softmax
from utils.cal_score import get_sentence_score
from torch.nn import AvgPool1d, MaxPool1d
from transformers import BertModel, BertConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer, KLDivLoss



class BiAttentionLayer(torch.nn.Module):
    """
    attention layer for NbestCrossBert
    every single hidden state representing one hypothesis, and they will attend to each other in this layer
    the hypothesis will attend to other hypothesis in the same N-best list, 
    so we need to re-design the encoder attention mask

           H11 H12 H13 H21 H22 H23
    H11     O   O   O   X   X   X
    H12     O   O   O   X   X   X
    H13     O   O   O   X   X   X
    H21     X   X   X   O   O   O
    H22     X   X   X   O   O   O
    H23     X   X   X   O   O   O
    """

    def __init__(self, input_dim , device, n_layers = 1):
        super().__init__()
        # self.config = BertConfig()
        self.input_dim = input_dim
        self.hidden_size = input_dim
        self.num_hidden_layers = n_layers
        # self.model = BertModel(self.config)

        print(f"cross Attend dim:{self.input_dim}, hidden state:{self.hidden_size}, Layers:{self.num_hidden_layers}")
        encoder_layer = TransformerEncoderLayer(
            d_model = 768, 
            nhead = 12, 
            dim_feedforward=3072, 
            layer_norm_eps = 1e-12, 
            batch_first = True,
            device = device
            )
        self.model = TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, inputs_embeds , encoder_attention_matrix):
        """
            input_states: (1, B, 768) 

            encoder_attention_matrix: (B,B)
        """
        # encoder_attention_matrix.masked_fill(encoder_attention_matrix == 0, float('-inf')).masked_fill(encoder_attention_matrix == 1, float(0.0))
        output = self.model(
            src = inputs_embeds,
            # token_type_ids = token_type_ids,
            # attention_mask = attention_mask,
            mask = encoder_attention_matrix
            )
        # print(f"output:{output.shape}")

        return output

    def config(self):
        return {
        "input_dim" : self.input_dim, 
        "n_layers" : self.num_hidden_layers,
        "heads": 12,
        "dim_feedforward": 3072,
        "layer_norm_eps": 1e-12
        }

class nBestCrossBert(torch.nn.Module):
    def __init__(
            self, 
            dataset, 
            device, 
            lstm_dim = 512, 
            use_fuseAttention = False,
            use_learnAttnWeight = False,
            addRes = False,
            fuseType = 'None',
            logSoftmax = 'False',
            lossType = 'KL'
        ):
        super().__init__()
        pretrain_name = getBertPretrainName(dataset)
        self.use_fuseAttention = use_fuseAttention
        self.use_learnAttnWeight = use_learnAttnWeight
        self.addRes = addRes
        self.fuseType = fuseType
        self.bert = BertModel.from_pretrained(pretrain_name)

        assert (fuseType in ['lstm', 'attn', 'none']), "fuseType must be 'lstm', 'attn', or 'none'"
        if (fuseType == 'lstm'):
            self.lstm = torch.nn.LSTM(
                input_size = 768,
                hidden_size = lstm_dim,
                num_layers = 2,
                dropout = 0.1,
                batch_first = True,
                bidirectional = True
                )

            self.concatLinear = torch.nn.Linear(2 * 2 * lstm_dim, 768)
        
        elif (fuseType == 'attn'):
            encoder_layer = TransformerEncoderLayer(
            d_model = 768, 
            nhead = 12, 
            dim_feedforward=3072, 
            layer_norm_eps = 1e-12, 
            batch_first = True,
            device = device
            )

            self.attnLayer = TransformerEncoder(encoder_layer, num_layers=1)
            self.concatLinear = torch.nn.Linear(2 * 768, 768)
        
        else:
            self.concatLinear = torch.nn.Linear(2 * 768, 768)

        self.fuseAttention = None
        self.clsConcatLinear = torch.nn.Linear(2 * 768, 768)
        self.finalLinear = torch.nn.Linear(770, 1)

        if (use_fuseAttention):
            self.clsConcatLinear = torch.nn.Linear(2 * 768, 766)
            self.fuseAttention = BiAttentionLayer(
                input_dim = 768,
                device = device
            )

            self.finalLinear = torch.nn.Linear(768, 1)
            
        self.device = device
        self.KL = False

        self.logSoftmax = logSoftmax
        if (lossType == 'KL'):
            self.activation_fn = torch.nn.LogSoftmax(dim = -1)
            self.KL = True
            self.loss = KLDivLoss(reduction='batchmean')
        else:
            self.activation_fn = torch.nn.Softmax(dim = -1)
            self.loss = torch.nn.CrossEntropyLoss()

        self.dropout = torch.nn.Dropout(p = 0.1)
        
    def forward(
            self, 
            input_ids, 
            attention_mask, 
            # batch_attention_mask, 
            batch_attention_matrix,
            am_score,
            ctc_score,
            labels = None, 
            N_best_index = None
        ):
        
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        cls = output.pooler_output # (B, 768)
        
        token_embeddings = output.last_hidden_state[:, 1:, :] # (B, L-1, 768)
        if (self.fuseType == 'lstm'):
            lstm_state, (h, c) = self.lstm(token_embeddings)
            # print(f'LSTM:{lstm_state.shape}')
            avg_pool = AvgPool1d(lstm_state.shape[1]).to(self.device)
            max_pool = MaxPool1d(lstm_state.shape[1]).to(self.device)

            avg_state = avg_pool(torch.transpose(lstm_state,1,2))
            max_state = max_pool(torch.transpose(lstm_state,1,2))

            avg_state = torch.transpose(avg_state, 1,2)
            max_state = torch.transpose(max_state, 1,2)
        
        elif (self.fuseType == 'attn'):
            attn_mask = attention_mask[:, 1:].clone()

            non_mask = (attn_mask == 1)
            mask_index = (attn_mask == 0)

            attn_mask[non_mask] = 0
            attn_mask[mask_index] = 1

            attn_state = self.attnLayer(
                src = token_embeddings,
                src_key_padding_mask = attn_mask
            )
            avg_pool = AvgPool1d(attn_state.shape[1]).to(self.device)
            max_pool = MaxPool1d(attn_state.shape[1]).to(self.device)

            avg_state = avg_pool(torch.transpose(attn_state,1,2))
            max_state = max_pool(torch.transpose(attn_state,1,2))

            avg_state = torch.transpose(avg_state, 1,2)
            max_state = torch.transpose(max_state, 1,2)

        else: # fuseType == 'none'
            avg_pool = AvgPool1d(token_embeddings.shape[1]).to(self.device)
            max_pool = MaxPool1d(token_embeddings.shape[1]).to(self.device)

            avg_state = avg_pool(torch.transpose(token_embeddings,1,2)).squeeze(-1)
            max_state = max_pool(torch.transpose(token_embeddings,1,2)).squeeze(-1)            
            # print(avg_state.shape)
            # print(max_state.shape)

        concat_state = torch.cat([avg_state, max_state], dim = -1)
        concatTrans = self.concatLinear(concat_state).squeeze(1)
        concatTrans = self.dropout(concatTrans)

        cls_concat_state = torch.cat([concatTrans, cls], dim = -1)
        clsConcatTrans = self.clsConcatLinear(cls_concat_state)
        concatTrans = self.dropout(concatTrans)

        clsConcatTrans = torch.cat([clsConcatTrans, am_score],  dim = -1)
        clsConcatTrans = torch.cat([clsConcatTrans, ctc_score],  dim = -1).float()

        if (self.fuseAttention is not None):
            clsConcatTrans = clsConcatTrans.unsqueeze(0) #(B, D) -> (1, B ,D) here we take B as length
            # batch_attention_matrix = batch_attention_matrix.unsqueeze(0)
            # print(clsConcatTrans.shape)
            # print(batch_attention_matrix)
            scoreAtt =self.fuseAttention(
                inputs_embeds = clsConcatTrans, 
                # token_type_ids = batch_token_type_id, 
                # attention_mask = batch_attention_mask,
                encoder_attention_matrix = batch_attention_matrix
            )
            if (self.addRes):
                scoreAtt = clsConcatTrans + scoreAtt
            finalScore = self.finalLinear(scoreAtt)

        else:
            finalScore = self.finalLinear(clsConcatTrans)
        
        # print(f'{finalScore.requires_grad}')
        finalScore = finalScore.squeeze(-1)
        if (self.fuseAttention):
            finalScore = finalScore.squeeze(0)
        logits = finalScore.clone()
        # print(f'logits.shape:{logits.shape}')

        loss = None
        if (labels is not None):
            assert(N_best_index is not None), "Must have N-best Index"
            start_index = 0
            for index in N_best_index:
                finalScore[start_index: start_index + index] = self.activation_fn(finalScore[start_index: start_index + index].clone())
                start_index += index
            # print(f"finalScore:{finalScore.shape}")
            # print(f"labels:{labels.shape}")
            if (self.KL):
                loss = self.loss(finalScore, labels)
            else: 
                loss = labels * torch.log(finalScore)
                loss = torch.neg(loss)
            # loss = self.loss(finalScore, labels)
        
        return {
            "loss": loss,
            "score": logits
        }
    def parameters(self):
        parameter = (
            list(self.bert.parameters()) + \
            list(self.concatLinear.parameters()) + \
            list(self.clsConcatLinear.parameters()) + \
            list(self.finalLinear.parameters())
        )
        if (self.use_fuseAttention):
            parameter = parameter + list(self.fuseAttention.model.parameters())
        
        if (hasattr(self, 'lstm')):
            parameter += self.lstm.parameters()
        if (hasattr(self, 'attnLayer')):
            parameter += self.attnLayer.parameters()
        
        return parameter
    
    def state_dict(self):
        fuseAttend = None
        if (self.use_fuseAttention):
            fuseAttend = self.fuseAttention.state_dict()

        state_dict = {
            "bert": self.bert.state_dict(),
            "concatLinear": self.concatLinear.state_dict(),
            "clsConcatLinear": self.clsConcatLinear.state_dict(),
            "finalLinear": self.finalLinear.state_dict(),
            "fuseAttend": fuseAttend
            }

        if hasattr(self, 'lstm'):
            state_dict['lstm'] = self.lstm.state_dict()
        
        if hasattr(self, 'attnLayer'):
            state_dict['attnLayer'] = self.attnLayer.state_dict()
        
        return state_dict
 
    def load_state_dict(self, checkpoint):
        self.bert.load_state_dict(checkpoint['bert'])
        if hasattr(self, 'lstm'):
            self.lstm.load_state_dict(checkpoint['lstm'])
        if hasattr(self, 'attnLayer'):
            self.attnLayer.load_state_dict(checkpoint['attnLayer'])
        self.concatLinear.load_state_dict(checkpoint['concatLinear'])
        self.clsConcatLinear.load_state_dict(checkpoint['clsConcatLinear'])
        self.finalLinear.load_state_dict(checkpoint['finalLinear'])
        if (checkpoint['fuseAttend'] is not None):
            self.fuseAttention.load_state_dict(checkpoint['fuseAttend'])


class pBert(torch.nn.Module):
    def __init__(self, dataset, device, hardLabel = False):
        super().__init__()
        pretrain_name = getBertPretrainName(dataset)
        self.bert = BertModel.from_pretrained(pretrain_name).to(device)
        self.linear = torch.nn.Linear(770, 1).to(device)

        self.hardLabel = hardLabel
        self.loss = torch.nn.KLDivLoss(reduction = 'batchmean')

        if (hardLabel):
            self.activation_fn = torch.nn.Softmax(dim = -1)
        else:
            self.activation_fn = torch.nn.LogSoftmax(dim = -1)

    def forward(
            self, 
            input_ids, 
            attention_mask, 
            N_best_index,
            am_score = None, 
            ctc_score = None, 
            labels = None
        ):
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).pooler_output

        # print(f"output:{output.shape}")

        clsConcatoutput = torch.cat([output, am_score], dim = -1)
        clsConcatoutput = torch.cat([clsConcatoutput, ctc_score], dim = -1)

        scores = self.linear(clsConcatoutput).squeeze(-1)
        final_score = scores.clone().detach()

        # print(f'scores:{scores.shape}')

        loss = None
        if (labels is not None):
            # print(f'labels.shape:{labels.shape}')
            start_index = 0
            for index in N_best_index:
                scores[start_index: start_index + index] = self.activation_fn(scores[start_index: start_index + index].clone())
                start_index += index
            
            if (self.hardLabel):
                loss = torch.sum(labels * torch.log(scores))
            else:
                loss = self.loss(scores, labels)
            # print(f'labels:{labels}')
            # print(f'scores:{scores}')
            # print(f"torch.log(scores):{torch.log(scores)}")
            # print(f'loss:{loss}')
        
                # loss = torch.neg(loss)

        return {
            "loss": loss,
            "score": final_score
        }

"""
Contrastive Learning: using marginalRanking loss
Sort the hypothesis by total ASR score and ranking first,  then calculate the score 
"""
import os
import sys
sys.path.append("../")
import torch
import torch.nn as nn
from transformers import BertModel
from src_utils.getPretrainName import getBertPretrainName
from utils.activation_function import SoftmaxOverNBest

class selfMarginLoss(nn.Module):
    def __init__(self, margin = 0.0):
        super().__init__()
        self.margin = margin
    def forward(self, scores, nBestIndex, werRank):
        # nBestIndex = a list of number of N-best e.x
        start_index = 0
        final_loss = torch.tensor(0.0, device = scores.device)

        for N, nBestRank in zip(nBestIndex, werRank): # i-th werRank
            for rank in nBestRank :
                compare = scores[start_index + rank + 1 : start_index + N] - scores[start_index + rank]
                # should be scores[rank] - scores[rank:] , so multiply -1
                compare = compare + self.margin

                result = compare[compare > 0]

                loss = result.sum()
                final_loss += loss

            start_index += N
        
        return final_loss

class ContrastBert(nn.Module):
    def __init__(self, args ,margin = 0.1):
        super().__init__()

        pretrain_name = getBertPretrainName(args['dataset'])
        self.bert = BertModel.from_pretrained(pretrain_name)
        self.linear = nn.Linear(770, 1)

        self.loss = selfMarginLoss(margin = margin)
        self.activation_fn = SoftmaxOverNBest()

    def forward(
            self,
            input_ids,
            attention_mask,
            am_score,
            ctc_score,
            nBestIndex,
            wer_rank,
            *args,
            **kwargs
    ):
        # input_ids, attention_mask , am_score and ctc_score are all sorted by WER and rank
        bert_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).pooler_output

        concat_state = torch.cat([bert_output, am_score], dim = -1)
        concat_state = torch.cat([concat_state, ctc_score], dim = -1)

        final_score = self.linear(concat_state)        
        # if (self.training):
        final_prob = self.activation_fn(final_score, nBestIndex)
        # Margin Loss
        loss = self.loss(final_prob, nBestIndex, wer_rank)


        return {
            'score': final_score,
            'loss': loss
        }

    def parameters(self):
        parameters = list(self.bert.parameters()) + list(self.linear.parameters())

        return parameters

    def state_dict(self):
        return{
            "bert": self.bert.state_dict(),
            "linear": self.linear.state_dict()
        }

    def load_state_dict(self, checkpoint):
        self.bert.load_state_dict(checkpoint['bert'])
        self.linear.load_state_dict(checkpoint['linear'])

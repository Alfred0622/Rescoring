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
class ContrastBert(nn.Module):
    def __init__(self, args ,margin = 0.1):
        self.super().__init__()

        pretrain_name = getBertPretrainName(args['dataset'])
        self.bert = BertModel.from_pretrained(pretrain_name)
        self.linear = nn.Linear(770, 1)

        self.loss = nn.MarginRankingLoss(margin = margin)
        self.activation_fn = SoftmaxOverNBest()

    def forward(
            self,
            input_ids,
            attention_mask,
            am_scores,
            ctc_scores,
            N_best_index,
            wer_Rank
    ):
        # input_ids, attention_mask , am_scores and ctc_scores are all sorted by WER and rank
        bert_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).pooler_output

        concat_state = torch.cat([bert_output, am_scores], dim = -1)
        concat_state = torch.cat([concat_state, ctc_scores], dim = -1)

        final_score = self.linear(concat_state)        
        if (self.training):
            start_index = 0
            totalLoss = torch.tensor(0.0)
            final_prob = self.activation_fn(final_score, N_best_index)
            for nBest in N_best_index:
                for index, rank in enumerate(wer_Rank):
                    pos_score = final_prob[start_index + rank:start_index + rank + 1].clone()
                    neg_score = final_prob[start_index + wer_Rank[index :]].clone()

                    pos_score = pos_score.expand_as(neg_score)
                    ones = torch.ones(pos_score.size()).to(final_prob.device)

                    totalLoss += self.loss(pos_score, neg_score, ones)
                
                start_index = start_index + nBest
        
        return {
            'logits': final_score,
            'loss': totalLoss
        }

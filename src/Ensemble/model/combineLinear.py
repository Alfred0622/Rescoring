import sys
sys.path.append("../")
import torch
import torch.nn as nn
from utils.activation_function import SoftmaxOverNBest


class combineLinear(nn.Module):
    def __init__(self, score_dict, reduction = 'mean'):
        self.feature_num = len(score_dict.keys())
        self.linear = nn.Linear(self.feature_num, 1)
        self.loss = nn.CrossEntropyLoss(reduction = reduction)
        self.activation_fn = SoftmaxOverNBest()
    
    def forward(self, scores, nBestIndex,ranks = None,labels = None):
        logits = self.linear(scores)
        logits = self.activation_fn(scores, nBestIndex)

        loss = self.loss(labels, logits)

        return {
            'logit': logits,
            'loss': loss
        }
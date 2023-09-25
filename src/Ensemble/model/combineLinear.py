from collections import OrderedDict
import sys

from torch import Tensor
sys.path.append("../")
import torch
import torch.nn as nn
from utils.activation_function import SoftmaxOverNBest


class combineLinear(nn.Module):
    def __init__(self, feature_num,bce = False, reduction = 'mean'):
        super().__init__()
        self.feature_num = feature_num
        print(f'feature_num:{self.feature_num}')
        self.linear = torch.nn.Sequential(
            nn.Linear(self.feature_num, 10 * self.feature_num),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(10 * self.feature_num, 20 * self.feature_num),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(20 * self.feature_num, 30 * self.feature_num),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            # nn.Linear(50 * self.feature_num, 100 * self.feature_num),
            # nn.ReLU(),
            # nn.Dropout(p = 0.2),
            # nn.Linear(100 * self.feature_num, 50 * self.feature_num),
            # nn.ReLU(),
            # nn.Dropout(p = 0.2),
            nn.Linear(30 * self.feature_num, 20 * self.feature_num),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(20 * self.feature_num, 10 * self.feature_num),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(10 * self.feature_num, 1)
        )
        self.reduction = reduction
        # self.loss = nn.CrossEntropyLoss(reduction = reduction, dim = -1)
        self.activation_fn = SoftmaxOverNBest()
        self.BCE = bce
        if self.BCE:
            self.activation_fn = torch.nn.Sigmoid()
            self.bce = torch.nn.BCELoss(reduction = self.reduction)
    
    def forward(self, scores, nBestIndex,ranks = None,labels = None):
        logits = self.linear(scores).squeeze(-1)


        if (labels is not None):
            
            if (self.BCE):
                # softmax_logits = self.activation_fn(logits)
                softmax_logits = self.activation_fn(logits, nBestIndex)
                loss = self.bce(softmax_logits, labels)

            else:
                softmax_logits = self.activation_fn(logits, nBestIndex)
                loss = labels * torch.log(softmax_logits)

                if (self.reduction == 'mean'):
                    loss = torch.mean(loss)
                elif (self.reduction == 'sum'):
                    loss = torch.sum(loss)
                loss = torch.neg(loss)
        else:
            loss = None

        return {
            'logit': logits,
            'loss': loss
        }

    def parameters(self):
        parameters = list(self.linear.parameters())
        return parameters
    
    def state_dict(self):
        return self.linear.state_dict()

    def load_state_dict(self, state_dict):
        self.linear.load_state_dict(state_dict)


class LinearSVM(nn.Module):
    def __init__(self, feature_num, C = 1.0, margin = 0.0):
        super().__init__()
        self.linear = nn.Linear(feature_num, 1)
        self.C = C
        self.margin = margin
        self.activation_fn = nn.Sigmoid()

    def forward(self, feature,nBestIndex ,labels = None, *args, **kwargs):
        score = self.linear(feature)
        logit = score.clone()
        if (labels is not None):
            score = self.activation_fn(score).squeeze(-1)
            labels[labels == 0] = -1
            # print(f'labels:{labels}')
            # print(f'labels:{labels.shape}')
            # print(f'score:{score.shape}')
            loss = torch.mean(torch.clamp(1 - labels * score, min=0))
            weight = self.linear.weight.squeeze()
            loss += self.C * (weight.t() @ weight) / 2.0
        else:
            loss = None

        return {
            'logit': logit,
            'loss': loss
        }

def prepare_CombineLinear(feature_num,bce = False ,reduction = 'mean'):
    model = combineLinear(feature_num, reduction= reduction, bce = False)
    return model

def prepare_LinearSVM(feature_num, C = 1.0, margin = 0.0):
    model = LinearSVM(feature_num = feature_num, C = C, margin = margin)
    return model    


import math

import torch
import logging

class PositionalEmbedding(torch.nn.Module):
    # get from  https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout)
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(0)
        pe = torch.zeros(1, max_len , d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
        
    def forward(self, x):
        x = x + self.pe[0, :x.size(1)]
        return self.dropout(x)
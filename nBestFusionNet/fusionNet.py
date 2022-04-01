import logging
import torch
import logging
import torch.nn as nn
from torch.nn.functional import log_softmax
from transformers import BertModel, BertTokenizer
from torch.nn import Conv1d

class fusionNet(nn.Module):
    def __init__(self, device, num_nBest, kernel_size = [2,3,4]):
        self.device = device
        self.num_nBest = num_nBest
        self.encoder = BertModel.from_pretrained('bert-base_chinese').to(device)
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels=768,out_channels=256, kernel_size = p, stride=1) for p in kernel_size
            ]
        )
        self.num_conv = len(self.conv)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256, self.num_nBest)
    def forward(self, input, seg, mask):
        batch_size = input.shape[0]

        output = self.encoder(input, seg, mask) 
        
        conv_output = []
        for conv in self.conv:
            conv_output.append(conv(output[0]))
        conv_output = torch.stack(conv_output) # conv * B * C_out * N
        
        conv_output = conv_output.view(1, batch_size, -1, self.num_nBest).squeeze(0)

        

        
    def recognize(self):
        pass
import logging
import torch
import numpy as np
import logging
import torch.nn as nn
from torch.nn.functional import log_softmax
from transformers import BertModel, BertTokenizer
from torch.nn import Conv1d, AvgPool1d, MaxPool1d
from torch.optim import Adam

class CNN(nn.Module):
    def __init__(
        self,
        device,
        nBest,
        kernel_size,
        stride = 1,
        pooling = 'max',
        pooling_kernel = 2
    ):
        nn.Module.__init__(self)
        self.device = device
        self.kernel_size = kernel_size
        self.nBest = nBest
        self.stride = stride
        fc_input_dim = np.floor(
            np.floor(
                ( (self.nBest - self.kernel_size) / self.stride) + 1
            ) - pooling_kernel + 1
        )

        self.model = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size = self.kernel_size, stride = self.stride), 
            nn.ReLU(), 
            MaxPool1d(kernel_size = pooling_kernel, stride = 1) if pooling == 'max' else AvgPool1d(
                kernel_size = pooling_kernel, stride = 1
            ),
            nn.Flatten(start_dim = 1),
            nn.Linear(256 *  int(fc_input_dim), 256)
        ).to(device)

    def forward(self, x):
        return self.model(x)

class fusionNet(nn.Module):
    def __init__(self,
    device, 
    num_nBest,
    kernel_size = [2,3,4],
    lr = 1e-4,
    max_lr = 0.02
    ):
        torch.nn.Module.__init__(self)
        self.device = device
        self.num_nBest = num_nBest
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.encoder = BertModel.from_pretrained('bert-base-chinese').to(device)
        self.conv = nn.ModuleList(
            [
                CNN(
                    self.device,
                    self.num_nBest,
                    k,
                ) for k in kernel_size
            ]
        )
        self.num_conv = len(self.conv)
        self.softmax = nn.Softmax(dim = -1)
        self.ce = nn.CrossEntropyLoss()

        self.fc = nn.Linear(256 * self.num_conv, self.num_nBest).to(device)
        
        model_parameters = list(self.encoder.parameters()) + list(self.fc.parameters())
        for c in self.conv:
            model_parameters += list(c.model.parameters())

        self.optimizer = Adam(model_parameters, lr = 1e-4)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer, 
            base_lr = lr, 
            max_lr = max_lr,
            cycle_momentum=False,
            step_size_up = 100,
            step_size_down = 100
        )

    def forward(self, input_id, mask, label):
        batch_size = int(input_id.shape[0] / self.num_nBest)

        output = self.encoder(
            input_ids = input_id,
            attention_mask = mask
        ) 
        
        output = output[0][:, 0, :]
        output = output.unsqueeze(0).view(batch_size, self.num_nBest, -1)
        output = torch.transpose(output, 1, 2)
        
        conv_output = []
        for i, conv in enumerate(self.conv):
            conv_output.append(conv(output))
        conv_output = torch.cat(conv_output, -1)

        conv_output = torch.flatten(conv_output, start_dim = 1).to(self.device) # flatten

        fc_output = self.fc(conv_output)

        loss = self.ce(fc_output, label)

        return loss
        
    def recognize(self, input_id, mask):
        
        input_id = input_id.squeeze(0)
        mask = mask.squeeze(0)
        output = self.encoder(
            input_ids = input_id,
            attention_mask = mask
        )
        
        output = output[0][:, 0, :]
        output = output.unsqueeze(0).view(1, self.num_nBest, -1)
        output = torch.transpose(output, 1, 2)
        
        conv_output = []
        for i, conv in enumerate(self.conv):
            conv_output.append(conv(output))
        conv_output = torch.cat(conv_output, -1)

        conv_output = torch.flatten(conv_output, start_dim = 1).to(self.device) # flatten

        fc_output = self.fc(conv_output)
        
        fc_output = self.softmax(fc_output)

        max_index = torch.argmax(fc_output)
        best_hyp = input_id[max_index].tolist()

        sep = best_hyp.index(102)
        return self.tokenizer.convert_ids_to_tokens(best_hyp[1:sep])

        

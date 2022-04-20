import logging
import torch
import logging
import torch.nn as nn
from torch.nn.functional import log_softmax
from transformers import BertModel, BertTokenizer
from torch.nn import Conv1d, AvgPool1d
from torch.optim import Adam

class fusionNet(nn.Module):
    def __init__(self,
    device, 
    num_nBest,
    kernel_size = [2,3,4],
    pooler_size = 4, 
    pooler_stride = 4,
    lr = 1e-4
    ):
        torch.nn.Module.__init__(self)
        self.device = device
        self.num_nBest = num_nBest
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.encoder = BertModel.from_pretrained('bert-base-chinese').to(device)
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels=768,out_channels=256, kernel_size = p, ).to(device) for p in kernel_size
            ]
        )
        self.num_conv = len(self.conv)
        # self.maxpool = nn.MaxPool1d()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)
        self.ce = nn.CrossEntropyLoss()
        self.pooler = AvgPool1d(pooler_size, stride = pooler_stride)

        self.kernel_size = torch.tensor(kernel_size)
        conv_dim = self.num_nBest - (self.kernel_size - 1)
        self.final_dim =  torch.floor((conv_dim - pooler_size) / pooler_stride + 1)
        
        total_dim = torch.sum(self.final_dim).long().item()
        self.fc = nn.Linear(256 * total_dim, self.num_nBest).to(device)
        
        
        model_parameters = list(self.encoder.parameters()) + list(self.fc.parameters())
        for c in self.conv:
            model_parameters += list(c.parameters())

        self.optimizer = Adam(model_parameters, lr = 1e-4)
        
        

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
            temp_output = self.relu(conv(output))
            temp_output = self.pooler(temp_output)
            conv_output.append(temp_output)
        conv_output = torch.cat(conv_output, -1)

        conv_output = torch.flatten(conv_output, start_dim = 1).to(self.device) # flatten

        fc_output = self.fc(conv_output)
        
        fc_output = self.softmax(fc_output)

        loss = self.ce(fc_output, label)

        return loss
        
    def recognize(self, input_id, mask):
        
        input_id = input_id.squeeze(0)
        mask = mask.squeeze(0)
        output = self.encoder(
            input_ids = input_id,
            attention_mask = mask
        )
        
        output = output[0][:, 0]
        output = output.unsqueeze(0).view(1, self.num_nBest, -1)
        output = torch.transpose(output, 1, 2)
        
        conv_output = []
        for i, conv in enumerate(self.conv):
            temp_output = self.relu(conv(output))
            temp_output = self.pooler(temp_output)
            conv_output.append(temp_output)
        conv_output = torch.cat(conv_output, -1)

        conv_output = torch.flatten(conv_output, start_dim = 1).to(self.device) # flatten

        fc_output = self.fc(conv_output)
        
        fc_output = self.softmax(fc_output)

        max_index = torch.argmax(fc_output)
        best_hyp = input_id[max_index].tolist()

        sep = best_hyp.index(102)
        return self.tokenizer.convert_ids_to_tokens(best_hyp[1:sep])

        

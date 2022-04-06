import logging
import torch
import logging
import torch.nn as nn
from torch.nn.functional import log_softmax
from transformers import BertModel, BertTokenizer
from torch.nn import Conv1d

class fusionNet(nn.Module):
    def __init__(self, device, num_nBest, kernel_size = [2,3,4], ):
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
        self.fc = nn.Linear(256, self.num_nBest).to(device)
        self.softmax = nn.Softmax()
        self.ce = nn.CrossEntropyLoss()

        

    def forward(self, input_id, seg, mask, label):
        batch_size = int(input_id.shape[0] / self.num_nBest)

        output = self.encoder(input_id, seg, mask) 
        
        logging.warning(f'output.shape before view:{output[0].shape}')
        
        output = output[0][:, 0]
        output = output.unsqueeze(0).view(batch_size, self.num_nBest, -1)
        output = torch.transpose(output, 1, 2)
        logging.warning(f'output.shape:{output.shape}')
        
        conv_output = []
        for i, conv in enumerate(self.conv):
            # batch_conv = []
            # for i in range(output.shape[0]):
            #     batch_conv.append(conv(output[i]))
            output = self.relu(conv(output))
            conv_output.append(output)
        
        conv_output = conv_output.view(batch_size, -1).to(self.device) # flatten
        logging.warning(f'conv_output.shape:{conv_output.shape}')
        
        fc_output = self.softmax(self.fc(conv_output), -1)
        logging.warning(f'fc.shape:{fc_output.shape}')

        loss = self.ce(fc_output, label)

        return loss
        
    def recognize(self, input_id, seg, mask):
        batch_size = int(input_id.shape[0] / self.num_nBest)

        output = self.encoder(input_id, seg, mask)
        
        output = output.view(batch_size, self.num_nBest, -1)
        
        conv_output = []
        for i, conv in enumerate(self.conv):
            conv_output.append(conv(output[0]))

            
        conv_output = torch.stack(conv_output) # conv * B * C_out * N
        
        conv_output = conv_output.view(batch_size, -1).to(self.device)
        
        fc_output = self.softmax(self.fc(conv_output), -1)

        max_index = torch.argmax(fc_output)
        best_hyp = input_id[max_index].tolist()

        sep = best_hyp.index(102)
        return self.tokenizer.convert_ids_to_tokens(best_hyp[1:sep])

        

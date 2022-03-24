import torch
import logging
from torch.nn.functional import log_softmax
from transformers import BertForMaskedLM, BertModel, BertTokenizer



class RescoreBert(torch.nn.Module):
    def __init__(self, device,weight = 0.2 ,use_MWER = False, use_MWED = False):
        torch.nn.Module.__init__(self)
        self.teacher = BertForMaskedLM.from_pretrained("bert-base-chinese").to(device)
        self.student = BertModel.from_pretrained("bert-base-chinese").to(device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.mask = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.weight = weight
        self.use_MWER = use_MWER
        self.use_MWED = use_MWED
        self.device = device
        self.l2_loss = torch.nn.MSELoss()

        self.fc = torch.nn.Linear(768,1)
    
    def adaption(self, input_id, segment, attention_mask):
        # random mask reference : https://github.com/jamescalam/transformers/blob/main/course/training/03_mlm_training.ipynb
        label = input_id
        rand = torch.rand(input_id.shape)
        mask_index = (rand < 0.15) * (input_id != 101) * (input_id != 102) * (input != 0)
        selection = []
        for i in range(input_id.shape[0]):
            selection.append(torch.flatten(mask_index[i].nonzero()).tolist())
        
        for i in range(input_id.shape[0]):
            input_id[i, selection[i]] = self.mask
        
        output = self.teacher(input_id, segment, attention_mask, labels = label)

        loss = output.loss

        return loss

    def forward(self, input_id, segment_id ,attention_mask, first_scores, cers):
        """
        input_id : (B, N_Best, Seq)
        segment_id : (B, Nbest, Seq)
        """
        # batch_size = input_id.shape[0]
        # nbest = input_id.shape[1]
        # input_id = input_id.view(batch_size * nbest, -1)
        # segment_id = segment_id.view(batch_size * nbest, -1)
        # attention_mask = attention_mask.view(batch_size * nbest, -1)

        # generate PLL loss from teacher
        pll_score = []  # (batch_size, N-Best)
        s_score = [] # (batch_size, N-Best)
        total_loss = 0.0

        loss_MWER = None
        loss_MWED = None

        for j in range(input_id.shape[0]): # for every hyp in this batch
            pll = 0.0
            token_len = torch.sum(input_id[j] != 0, dim = -1)
            seg = segment_id[j].unsqueeze(0)
            mask = attention_mask[j].unsqueeze(0)
            tmp = input_id[j].unsqueeze(0)
            for k in range(1 , token_len - 1): # for each token in this hyp (exclude padding)
                mask_token = tmp[0][k].item() 
                tmp[0][k] = self.mask
                tmp = tmp.to(self.device)

                outputs = self.teacher(tmp, seg, mask)
                prediction = torch.log_softmax(outputs[0], -1)
                
                pll += prediction[0, k , mask_token]
                tmp[0][k] = mask_token

            pll_score.append(pll)
        
        pll_score = torch.tensor(pll_score)

        s_output = self.student(input_id, segment_id, attention_mask)
        s_score = self.fc(s_output[0][:, 0]).view(pll_score.shape)

        total_score = self.l2_loss(pll_score, s_score)
        total_loss = total_score.sum()

        weight_sum = first_scores + s_score
        
        # MWER
        if (self.use_MWER):
            wer = torch.stack(
                [
                    ((s[1] + s[2] + s[3]) / (s[0] + s[1] + s[2])) for s in cers
                ]
            )
            p_hyp = torch.softmax(torch.neg(torch.tensor(weight_sum)), dim = -1)
            avg_error = torch.mean(wer)
            avg_error = torch.full(wer.shape, avg_error)
            
            loss_MWER = p_hyp * (wer - avg_error)
            loss_MWER = loss_MWER.sum()
            
            total_loss += loss_MWER
        
        # MWED
        elif (self.use_MWED):
            wer = torch.stack(
                [
                    ((s[1] + s[2] + s[3]) / (s[0] + s[1] + s[2])) for s in cers
                ]
            )
            d_error = torch.softmax(wer, dim = -1)
            d_score = torch.softmax(s_score, dim = -1)

            loss_MWED = d_error * torch.log(d_score)
            loss_MWED = -(loss_MWED.sum())
            
            total_loss += loss_MWED
        return total_loss

    def recognize(self, input_id, segment_id, attention_mask, first_scores):
        output = self.student(input_id, segment_id, attention_mask)
        rescore = self.fc(output[0][:, 0]).view(first_scores.shape)
        

        max_sentence = torch.argmax(first_scores + rescore)

        return input_id[max_sentence]
    
    
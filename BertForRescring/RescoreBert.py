import torch
import logging
from torch.nn.functional import log_softmax
from transformers import BertForMaskedLM, BertModel, BertTokenizer



class RescoreBert(torch.nn.Module):
    def __init__(self, device,weight = 0.2 ,use_MWER = False, use_MWED = False):
        torch.nn.Module.__init__(self)
        self.teacher = BertForMaskedLM.from_pretrain("bert-base-chinese")
        self.student = BertModel.from_pretrain("bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrain("bert-base_chinese")
        self.mask = self.tokenizer.convert_tokens_to_ids["<MASK>"]
        self.weight = weight
        self.use_MWER = use_MWER
        self.use_MWED = use_MWED
        self.device = device

    def forward(self, input_id, segment_id ,attention_mask, first_scores, cers):
        """
        input_id : (B, N_Best, Seq)
        segment_id : (B, Nbest, Seq)
        """
        batch_size = input_id.shape[0]
        nbest = input_id.shape[1]
        input_id = input_id.view(batch_size * nbest, -1)
        segment_id = segment_id.view(batch_size * nbest, -1)
        attention_mask = attention_mask.view(batch_size * nbest, -1)

        # generate PLL loss from teacher
        pll_score = []  # (batch_size, N-Best)
        s_score = [] # (batch_size, N-Best)
        total_loss = 0.0

        loss_MWER = None
        loss_MWED = None

        temp_pll = []
        for j in range(input_id.shape[0]): # for every hyp in this batch
            pll = 0.0
            for k in range(1, input_id.shape[1] - 1): # for each token in this hyp
                mask_token = input_id[j][k]
                tmp = input_id[j][:k] + self.mask + input_id[j][k+1:]
                tmp = torch.tensor(tmp).to(self.device)
                    
                outputs = self.teacher(tmp, token_type_id = segment_id)
                prediction = outputs[0]
                pll += torch.log_softmax(prediction[0, mask_token], -1)
            temp_pll.append(pll)
        pll_score.append(temp_pll)
        
        pll_score = torch.tensor(pll_score)

        s_output = self.student(input_id, segment_id, attention_mask)
        s_score = log_softmax(s_output[0][0], dim = -1)

        total_loss = torch.sub(s_score , pll_score).sum()

        weight_sum = first_scores + s_score

        # MWER
        if (self.use_MWER):
            p_hyp = torch.softmax(torch.neg(torch.tensor(weight_sum)), dim = -1)
            error_hyp = [] # need to get the data of CER , shape = (Batch, N-Best)
            avg_error = torch.full(torch.mean(cers))
            
            loss_MWER = p_hyp * torch.sub(cers , avg_error).sum()
            
            total_loss += loss_MWER
        
        # MWED
        elif (self.use_MWED):
            d_error = torch.softmax(torch.tensor(cers), dim = -1) # need to get cer tensor
            d_score = torch.softmax(torch.tensor(s_score), dim = -1)

            loss_MWED = torch.nn.CrossEntropyLoss(d_error, d_score)
            
            total_loss += loss_MWED
        return total_loss

    def recognize(self, input_id, segment_id, attention_mask, first_scores):
        output = self.student(input_id, segment_id, attention_mask)
        rescore = output[0]

        max_sentence = torch.argmax(rescore + first_scores)
        return input_id[max_sentence]

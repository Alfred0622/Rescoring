import torch
import torch.nn as nn
import logging
from torch.nn.functional import log_softmax
from transformers import (
    BertForMaskedLM,
    AutoModelForCausalLM,
    BertTokenizerFast,

)
from torch.optim import AdamW
from utils.cal_score import get_sentence_score

class CLMRescorer(nn.Module):
    def __init__(self, device, lr = 5e-4, mode = 'scoring'):
        nn.Module.__init__(self)
        self.model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese').to(device)
        self.lr = float(lr)
        self.optimizer = AdamW(self.model.parameters(), lr = self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.mode = mode # 'scoring' or 'generate' 
    def forward(self, input_ids, labels, first_score = None, cers = None):
        # first_score, cer will not be None during validation
        if (not self.training):
            output = self.model(input_ids, labels = labels)
            logit = output.logits
            score = get_sentence_score(logit, input_ids)
            score = score.clone().detach().cpu()

            weighted_score = first_score + score
            best_hyp = torch.argmax(weighted_score)

            return output.loss,  cers[best_hyp]
        else:
            output = self.model(input_ids, labels = labels)
            return output.loss


    def recognize(self, input_ids):
        output = self.model(input_ids)
        
        # return output score
        logit = output.logits # (B, pad_seq, vocab)
        score = get_sentence_score(logit, input_ids)
        return score
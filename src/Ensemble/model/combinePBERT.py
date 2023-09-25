import sys
sys.path.append("../")
import torch
import torch.nn as nn
from src_utils.getPretrainName import getBertPretrainName
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer
)
from utils.activation_function import SoftmaxOverNBest

class ensemble_pBert(torch.nn.Module):
    def __init__(
        self,
        args,
        train_args,
        feature_num,
        device,
        output_attention=True,
    ):
        super().__init__()
        pretrain_name = getBertPretrainName(args["dataset"])
        config = BertConfig.from_pretrained(pretrain_name)
        config.output_attentions = True
        config.output_hidden_states = True

        self.feature_num = feature_num
        self.bert = BertModel(config=config).from_pretrained(pretrain_name).to(device)
        self.linear = torch.nn.Linear(768 + self.feature_num, 1).to(device)

        self.hardLabel = train_args["hard_label"]
        self.loss_type = train_args["loss_type"]
        self.loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.BCE = torch.nn.BCELoss()

        self.dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)

        self.output_attention = output_attention

        self.reduction = train_args["reduction"]

        self.activation_fn = SoftmaxOverNBest()

        # print(f"output_attention:{self.output_attention}")
        # print(f"weightByWER:{self.weightByWER}")

    def forward(
        self,
        input_ids,
        attention_mask,
        nBestIndex,
        score_features,
        labels=None,
        *args,
        **kwargs,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attention,
        )

        output = bert_output.pooler_output
        output = self.dropout(output)

        clsConcatoutput = torch.cat([output, score_features], dim=-1)

        scores = self.linear(clsConcatoutput).squeeze(-1)
        # print(f'scores:{scores}')
        final_score = scores.clone().detach()

        loss = None
        if labels is not None:
            # print(f'labels:{labels}')
            if self.hardLabel:
                scores = self.activation_fn(scores, nBestIndex, log_score=False)
                if self.loss_type == "Entropy":
                    loss = labels * torch.log(scores)
                    loss = torch.neg(loss)
                elif self.loss_type == "BCE":
                    loss = self.BCE(scores, labels)
            else:
                if self.loss_type == "KL":
                    scores = self.activation_fn(
                        scores, nBestIndex, log_score=True
                    )  # Log_Softmax
                    loss = self.loss(scores, labels)
                else:
                    scores = self.activation_fn(scores, nBestIndex, log_score=False)
                    loss = labels * torch.log(scores)
                    loss = torch.neg(loss)

            if self.reduction == "sum":
                loss = torch.sum(loss)
            elif self.reduction == "mean":
                loss = torch.mean(loss)

        return {
            "loss": loss,
            "logit": final_score,
        }

def prepare_ensemble_pbert(config, train_args, device, feature_num, *args, **kwargs):
    pretrain_name = getBertPretrainName(config['dataset'])
    tokenizer = BertTokenizer.from_pretrained(pretrain_name)
    
    model = ensemble_pBert(
        args = config,
        train_args= train_args,
        feature_num=feature_num,
        device = device
    )

    return model, tokenizer
"""
Contrastive Learning: using marginalRanking loss
Sort the hypothesis by total ASR score and ranking first,  then calculate the score 
"""
import os
import sys

sys.path.append("../")
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertLayer
from src_utils.getPretrainName import getBertPretrainName
from utils.activation_function import SoftmaxOverNBest
from utils.Pooling import AvgPooling, MaxPooling


class selfMarginLoss(nn.Module):
    def __init__(self, margin=0.0, reduction="mean", useTopOnly=False):
        super().__init__()
        self.margin = margin
        assert reduction in ["mean", "sum"], "reduction must in ['mean', 'sum']"
        self.reduction = reduction
        self.top_only = useTopOnly

    def forward(self, scores, nBestIndex, werRank):
        # nBestIndex = a list of number of N-best e.x
        start_index = 0
        final_loss = torch.tensor([0.0], device=scores.device, dtype=torch.float64)

        for N, nBestRank in zip(nBestIndex, werRank):  # i-th werRank ex.[50, 50, 50]
            nBestRank = nBestRank.to(scores.device)
            if self.top_only:
                compare = (
                    scores[start_index + nBestRank[1:]]  # x2
                    - scores[start_index + nBestRank[0]]  # x1
                )

                compare = compare + self.margin

                result = compare[compare > 0]

                loss = result.sum()

                if self.reduction == "mean":
                    loss = loss / len(nBestRank[1:])
                final_loss += loss

            else:
                print(f"score:{scores}")
                print(f"index:{start_index}:{start_index + N}")
                nBestScore = scores[start_index : start_index + N]
                print(f"nBestScore:{nBestScore}")
                for i, rank in enumerate(nBestRank[:-1]):
                    print(f"rank:{rank}")
                    print(f"pos:{nBestScore[rank]}")
                    print(f"neg:{nBestScore[nBestRank[i + 1 :]]}")
                    compare = nBestScore[nBestRank[i + 1 :]] - nBestScore[rank]
                    compare = compare + self.margin
                    result = compare[compare > 0]

                    loss = result.sum()
                    if self.reduction == "mean":
                        loss = loss / len(nBestRank[i + 1 :])

                    final_loss += loss

                start_index += N

        # for i, rank in enumerate(nBestRank[:-1]):
        #     compare = (
        #         scores[start_index + nBestRank[i + 1 :]]  # x2
        #         - scores[start_index + rank]  # x1
        #     )  # take every index after the present rank
        #     # should be scores[rank] - scores[rank:] , so multiply -1
        #     compare = compare + self.margin

        #     result = compare[compare > 0]

        #     loss = result.sum()
        #     if self.reduction == "mean":
        #         loss = loss / len(scores[start_index + nBestRank[i + 1 :]])

        #     final_loss += loss

        # start_index += N

        return final_loss


class torchMarginLoss(nn.Module):
    def __init__(self, margin=0.0, reduction="mean", useTopOnly=False):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.useTopOnly = useTopOnly
        self.loss = torch.nn.MarginRankingLoss(margin=margin, reduction=reduction)

    def forward(self, scores, nBestIndex, werRank):
        start_index = 0
        total_loss = torch.tensor([0.0], device=scores.device, dtype=torch.float64)

        for N, nBestRank in zip(nBestIndex, werRank):

            nBestScore = scores[start_index : start_index + N]
            print(f"score:{scores}")
            print(f"index:{start_index}:{start_index + N}")
            print(f"nBestScore:{nBestScore}")

            for j, rank in enumerate(nBestRank[:-1]):
                print(f"rank:{rank}")
                print(f"pos:{nBestScore[rank]}")

                neg_score = nBestScore[nBestRank[j + 1 :]]

                print(f"neg:{neg_score}")

                pos_score = nBestScore[rank].expand_as(neg_score)

                ones = torch.ones(pos_score.shape).to(scores.device)

                loss = self.loss(pos_score, neg_score, ones)
                # if self.reduction == "mean":
                #     loss = loss / neg_score.shape[0]
                # print(f"loss:{loss}")

                total_loss += loss
            start_index += N

        return total_loss


class marginalBert(nn.Module):
    def __init__(self, args, margin=0.1, useTopOnly=False, useTorch=False):
        super().__init__()

        pretrain_name = getBertPretrainName(args["dataset"])
        self.bert = BertModel.from_pretrained(pretrain_name)
        self.linear = nn.Linear(770, 1)
        if useTorch:
            self.loss = torchMarginLoss(margin=margin)
        else:
            self.loss = selfMarginLoss(margin=margin, useTopOnly=useTopOnly)
        self.activation_fn = SoftmaxOverNBest()

        print(f"useTorch:{useTorch}")

    def forward(
        self,
        input_ids,
        attention_mask,
        am_score,
        ctc_score,
        nBestIndex,
        wer_rank,
        labels,
        extra_loss=False,
        *args,
        **kwargs,
    ):
        for wer in wer_rank:
            if len(wer) < 0:
                print(wer)
        # input_ids, attention_mask , am_score and ctc_score are all sorted by WER and rank
        bert_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output

        concat_state = torch.cat([bert_output, am_score], dim=-1)
        concat_state = torch.cat([concat_state, ctc_score], dim=-1)

        final_score = self.linear(concat_state).squeeze(-1)
        # print(f"final_score:{final_score}")
        # if (self.training):
        final_prob = self.activation_fn(final_score, nBestIndex)
        ce_loss = labels * torch.log(final_prob)
        ce_loss = torch.neg(torch.sum(ce_loss)) / final_score.shape[0]

        # Margin Loss
        if self.training and extra_loss:
            margin_loss = self.loss(final_prob, nBestIndex, wer_rank)

            loss = ce_loss + margin_loss
        else:
            loss = ce_loss
            margin_loss = None

        return {
            "score": final_score,
            "loss": loss,
            "CE_loss": ce_loss,
            "contrast_loss": margin_loss,
        }

    def parameters(self):
        parameters = list(self.bert.parameters()) + list(self.linear.parameters())

        return parameters

    def state_dict(self):
        return {"bert": self.bert.state_dict(), "linear": self.linear.state_dict()}

    def load_state_dict(self, checkpoint):
        self.bert.load_state_dict(checkpoint["bert"])
        self.linear.load_state_dict(checkpoint["linear"])


class contrastBert(nn.Module):
    def __init__(self, args, train_args):
        super().__init__()

        pretrain_name = getBertPretrainName(args["dataset"])
        self.bert = BertModel.from_pretrained(pretrain_name)
        self.linear = nn.Linear(770, 1)
        self.contrast_weight = torch.tensor(train_args["contrast_weight"]).float()

        self.activation_fn = SoftmaxOverNBest()
        self.useTopOnly = train_args["useTopOnly"]
        self.compare = train_args["compareWith"].strip().upper()

        # self.cosSim = nn.CosineSimilarity(dim=-1)
        # self.cosSim = torch.tensordot(dims=-1)
        self.pooling = MaxPooling(noCLS=train_args["noCLS"], noSEP=train_args["noSEP"])
        self.layerOp = train_args["layer_op"]
        if self.layerOp is not None and "lastInit" in self.layerOp:
            config = BertConfig.from_pretrained(pretrain_name)
            layer_init = self.layerOp.split("_")[-1]
            print(f"init Layer:{layer_init}")
            for i in range(1, int(layer_init) + 1):
                self.bert.encoder.layer[-i] = BertLayer(config)

    def forward(
        self,
        input_ids,
        attention_mask,
        am_score,
        ctc_score,
        nBestIndex,
        labels,
        ref_ids,
        ref_mask,
        extra_loss=True,
        *args,
        **kwargs,
    ):
        # input_ids, attention_mask , am_score and ctc_score are all sorted by WER and rank
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        bert_output = output.pooler_output  # (B, 768)
        bert_embedding = output.last_hidden_state
        # (B, 768)

        concat_state = torch.cat([bert_output, am_score], dim=-1)
        concat_state = torch.cat([concat_state, ctc_score], dim=-1)

        final_score = self.linear(concat_state).squeeze(-1)
        final_prob = self.activation_fn(final_score, nBestIndex)
        ce_loss = labels * torch.log(final_prob)
        ce_loss = torch.neg(torch.sum(ce_loss))

        if extra_loss and self.training:
            if self.compare == "TOP":
                sim_matrix = torch.tensordot(bert_output, bert_output, dims=([1], [1]))
                norm = torch.norm(bert_output, dim=-1)
                # print(f"norm:{norm.shape}")
                sim_matrix = sim_matrix / norm
                sim_matrix = sim_matrix / norm.unsqueeze(-1)

                # print(f"sim_matrix.shape:{sim_matrix}")
                # print(f"diag sim:{torch.diagonal(sim_matrix, 0)}")
                sim_matrix = self.activation_fn(sim_matrix, nBestIndex)
                # print(f"diag sim agter softmax:{torch.diagonal(sim_matrix, 0)}")
                top_index = labels == 1
                sim_value = torch.diagonal(sim_matrix, 0)
                top_sim = sim_value[top_index]
                contrastLoss = torch.neg(torch.sum(torch.log(top_sim)))

            elif self.compare == "POOL":
                pooled_embedding = self.pooling(bert_embedding, attention_mask)
                sim_matrix = self.cosSim(bert_output, pooled_embedding)
                sim_matrix = self.activation_fn(sim_matrix, nBestIndex)
                sim_value = torch.diag(sim_matrix, 0)
                contrastLoss = torch.neg(torch.sum(sim_value))

            elif self.compare == "REF":
                ref_output = self.bert(
                    input_ids=ref_ids, attention_mask=ref_mask
                ).pooler_output
                sim_matrix = torch.tensordot(ref_output, bert_output, dims=([1], [1]))

                sim_matrix = self.activation_fn(sim_matrix, nBestIndex)
                top_index = (labels == 1).nonzero()

                print(f"sim_matrix after softmax:{sim_matrix}")

                # print(f"top_index:{top_index}")
                contrastLoss = torch.tensor([0.0]).cuda()
                for i, batch_top in enumerate(top_index):
                    print(f"sim:{sim_matrix[i][batch_top]}")
                    print(f"loss:{torch.log(sim_matrix[i][batch_top])}")
                    contrastLoss += torch.log(sim_matrix[i][batch_top])
                contrastLoss = torch.neg(contrastLoss)
                print(f"contrastLoss:{contrastLoss}")

            # contrastLoss = contrastLoss  # batch_mean
            loss = ce_loss + self.contrast_weight * contrastLoss
        else:
            loss = ce_loss
            contrastLoss = None

        return {
            "score": final_score,
            "loss": loss,
            "CE_loss": ce_loss,
            "contrast_loss": contrastLoss,
        }

    def parameters(self):
        parameters = list(self.bert.parameters()) + list(self.linear.parameters())

        return parameters

    def state_dict(self):
        return {"bert": self.bert.state_dict(), "linear": self.linear.state_dict()}

    def load_state_dict(self, checkpoint):
        self.bert.load_state_dict(checkpoint["bert"])
        self.linear.load_state_dict(checkpoint["linear"])

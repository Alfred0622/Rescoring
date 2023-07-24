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
import re


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
                # print(f"score:{scores}")
                # print(f"index:{start_index}:{start_index + N}")
                nBestScore = scores[start_index : start_index + N]
                # print(f"nBestScore:{nBestScore}")
                for i, rank in enumerate(nBestRank[:-1]):
                    # print(f"rank:{rank}")
                    # print(f"pos:{nBestScore[rank]}")
                    # print(f"neg:{nBestScore[nBestRank[i + 1 :]]}")
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

        if self.useTopOnly:
            for N, nBestRank in zip(nBestIndex, werRank):
                nBestScore = scores[start_index : start_index + N]
                neg_score = nBestScore[nBestRank[1:]]
                pos_score = nBestScore[0].expand_as(neg_score)
                ones = torch.ones(pos_score.shape).to(scores.device)

                loss = self.loss(pos_score, neg_score, ones)

                total_loss += loss

                start_index += N
        else:
            for N, nBestRank in zip(nBestIndex, werRank):
                nBestScore = scores[start_index : start_index + N]

                for j, rank in enumerate(nBestRank[:-1]):
                    # print(f"rank:{rank}")
                    # print(f"pos:{nBestScore[rank]}")

                    neg_score = nBestScore[nBestRank[j + 1 :]]

                    # print(f"neg:{neg_score}")

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
    def __init__(self, args, train_args, margin=0.1, useTopOnly=False, useTorch=False):
        super().__init__()

        pretrain_name = getBertPretrainName(args["dataset"])
        self.bert = BertModel.from_pretrained(pretrain_name)
        print(f"dropout_rate:{self.bert.config.hidden_dropout_prob}")
        self.linear = torch.nn.Sequential(
            nn.Dropout(self.bert.config.hidden_dropout_prob), nn.Linear(770, 1)
        )
        self.reduction = train_args["reduction"]
        if useTorch:
            self.loss = torchMarginLoss(margin=margin, reduction=self.reduction)
        else:
            self.loss = selfMarginLoss(
                margin=margin, useTopOnly=useTopOnly, reduction=self.reduction
            )
        self.activation_fn = SoftmaxOverNBest()

        self.layerOp = train_args["layer_op"]

        print(f"useTorch:{useTorch}")

        self.layerOp = train_args["layer_op"]
        if self.layerOp is not None:
            if "lastInit" in self.layerOp:
                config = BertConfig.from_pretrained(pretrain_name)
                layer_init = self.layerOp.split("_")[-1]
                print(f"init Layer:{layer_init}")
                for i in range(1, int(layer_init) + 1):
                    self.bert.encoder.layer[-i] = BertLayer(config)
            elif self.layerOp == "LSTM":
                self.LSTM = torch.nn.LSTM(
                    input_size=768,
                    hidden_size=1024,
                    batch_first=True,
                    bidirectional=True,
                    proj_size=768,
                )
                self.concatLSTM = torch.nn.Sequential(
                    nn.Dropout(p=0.1), nn.Linear(2 * 768, 768), nn.ReLU()
                )

                self.concatCLSLinear = torch.nn.Sequential(
                    nn.Dropout(p=0.1), nn.Linear(2 * 768, 768), nn.ReLU()
                )

                self.maxPool = MaxPooling(
                    noCLS=train_args["noCLS"], noSEP=train_args["noSEP"]
                )
                self.avgPool = AvgPooling(
                    noCLS=train_args["noCLS"], noSEP=train_args["noSEP"]
                )

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
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = output.pooler_output

        if self.layerOp == "LSTM":
            bert_embedding = output.last_hidden_state
            attn_mask = attention_mask.clone()
            max_pool = self.maxPool(bert_embedding, attn_mask)
            avg_pool = self.avgPool(bert_embedding, attn_mask)
            # concat pooled LSTM
            concat_state = torch.cat([max_pool, avg_pool], dim=-1)
            concat_state = self.concatLSTM(concat_state)
            # concat CLS
            # print(f"cls:{bert_output.shape}")
            # print(f"concat_state:{concat_state.shape}")
            concat_state = torch.cat([bert_output, concat_state], dim=-1)
            bert_output = self.concatCLSLinear(concat_state)

        concat_state = torch.cat([bert_output, am_score], dim=-1)
        concat_state = torch.cat([concat_state, ctc_score], dim=-1)

        final_score = self.linear(concat_state).squeeze(-1)
        # print(f"final_score:{final_score}")
        # if (self.training):
        final_prob = self.activation_fn(final_score, nBestIndex)
        ce_loss = labels * torch.log(final_prob)
        if self.reduction == "sum":
            ce_loss = torch.neg(torch.sum(ce_loss))
        if self.reduction == "mean":
            ce_loss = torch.neg(torch.mean(ce_loss))

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
        checkpoint = {
            "bert": self.bert.state_dict(),
            "linear": self.linear.state_dict(),
        }
        if self.layerOp is not None and "LSTM" in self.compare:
            checkpoint["LSTM"] = self.LSTM.state_dict()
            checkpoint["concatLSTM"] = self.concatLSTM.state_dict()
            checkpoint["concatCLSLinear"] = self.concatCLSLinear.state_dict()

        return checkpoint

    def load_state_dict(self, checkpoint):
        self.bert.load_state_dict(checkpoint["bert"])
        self.linear.load_state_dict(checkpoint["linear"])
        if self.layerOp is not None and self.layerOp == "LSTM":
            self.LSTM.load_state_dict(checkpoint["LSTM"])
            self.concatLSTM.load_state_dict(checkpoint["concatLSTM"])
            self.concatCLSLinear.load_state_dict(checkpoint["concatCLSLinear"])


class contrastBert(nn.Module):
    def __init__(self, args, train_args):
        super().__init__()

        pretrain_name = getBertPretrainName(args["dataset"])
        self.bert = BertModel.from_pretrained(pretrain_name)
        self.linear = torch.nn.Sequential(
            torch.nn.Dropout(self.bert.config.hidden_dropout_prob), nn.Linear(770, 1)
        )  # final Linear
        self.contrast_weight = torch.tensor(train_args["contrast_weight"]).float()

        self.loss_type = train_args["loss_type"]
        self.activation_fn = SoftmaxOverNBest()
        self.useTopOnly = train_args["useTopOnly"]
        self.compare = train_args["compareWith"].strip().upper()

        self.reduction = train_args["reduction"]

        self.temperature = float(train_args["temperature"])

        self.BCE = torch.nn.BCELoss(reduction=self.reduction)
        self.CE = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.softmax = torch.nn.Softmax(dim=-1)

        # self.cosSim = nn.CosineSimilarity(dim=-1)
        # self.cosSim = torch.tensordot(dims=-1)
        self.pooling = MaxPooling(noCLS=train_args["noCLS"], noSEP=train_args["noSEP"])
        self.layerOp = train_args["layer_op"]
        if self.layerOp is not None:
            if "lastInit" in self.layerOp:
                config = BertConfig.from_pretrained(pretrain_name)
                layer_init = self.layerOp.split("_")[-1]
                print(f"init Layer:{layer_init}")
                for i in range(1, int(layer_init) + 1):
                    self.bert.encoder.layer[-i] = BertLayer(config)
        if "LSTM" in self.compare:
            self.LSTM = torch.nn.LSTM(
                input_size=768,
                hidden_size=1024,
                batch_first=True,
                num_layers=2,
                dropout=0.1,
                bidirectional=True,
                proj_size=768,
            )
            self.concatLSTM = torch.nn.Sequential(
                nn.Dropout(p=0.1), nn.Linear(2 * 2 * 768, 768), nn.ReLU()
            )
            self.pretrain_classifier = torch.nn.Sequential(
                nn.Dropout(p=0.1), nn.Linear(2 * 2 * 768, 1), nn.Sigmoid()
            )

            self.LSTMLMHead = torch.nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(2 * 768, self.bert.config.vocab_size),
                nn.ReLU(),
            )

            self.maxPool = MaxPooling(
                noCLS=train_args["noCLS"], noSEP=train_args["noSEP"]
            )
            self.avgPool = AvgPooling(
                noCLS=train_args["noCLS"], noSEP=train_args["noSEP"]
            )

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
        tokenizer=None,
        extra_loss=True,
        *args,
        **kwargs,
    ):
        # input_ids, attention_mask , am_score and ctc_score are all sorted by WER and rank
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        bert_output = output.pooler_output  # (B, 768)

        # (B, 768)
        concat_state = torch.cat([bert_output, am_score], dim=-1)
        concat_state = torch.cat([concat_state, ctc_score], dim=-1)

        final_score = self.linear(concat_state).squeeze(-1)
        final_prob = self.activation_fn(final_score, nBestIndex)
        if self.loss_type == "BCE":
            ce_loss = self.BCE(final_prob, labels)
        else:
            ce_loss = labels * torch.log(final_prob)
            if self.reduction == "sum":
                ce_loss = torch.neg(torch.sum(ce_loss))
            if self.reduction == "mean":
                ce_loss = torch.neg(torch.mean(ce_loss))

        if extra_loss and self.training:
            if self.compare == "SELF":
                sim_matrix = torch.tensordot(bert_output, bert_output, dims=([1], [1]))
                norm = torch.norm(bert_output, dim=-1)
                sim_matrix = sim_matrix / norm
                sim_matrix = sim_matrix / norm.unsqueeze(-1)

                sim_matrix = sim_matrix / self.temperature
                sim_matrix = self.activation_fn(sim_matrix, nBestIndex)
                sim_value = torch.diagonal(sim_matrix, 0)
                if self.useTopOnly:
                    top_index = labels == 1
                    top_sim = sim_value[top_index]
                    if self.reduction == "sum":
                        contrastLoss = torch.neg(torch.sum(torch.log(top_sim)))
                    elif self.reduction == "mean":
                        contrastLoss = torch.neg(torch.mean(torch.log(top_sim)))
                else:
                    if self.reduction == "sum":
                        contrastLoss = torch.neg(torch.sum(torch.log(sim_value)))
                    elif self.reduction == "mean":
                        contrastLoss = torch.neg(torch.mean(torch.log(sim_value)))

            elif self.compare in ["SELF-QE", "SELF-QE-HARD"]:
                # print(f"input_ids:{input_ids}")
                qe_index = input_ids == tokenizer.convert_tokens_to_ids("[MASK]")
                # print(f"qe_index:{qe_index}")
                qe_output = output.last_hidden_state[qe_index, :]
                # print(f"qe_output:{qe_output.shape}")

                sim_matrix = torch.tensordot(bert_output, qe_output, dims=([1], [1]))
                cls_norm = torch.norm(bert_output, dim=-1)
                qe_norm = torch.norm(qe_output, dim=-1)

                sim_matrix = (sim_matrix / qe_norm) / cls_norm.unsqueeze(-1)
                sim_matrix = sim_matrix / self.temperature
                if self.compare == "SELF-QE":
                    sim_matrix = self.activation_fn(sim_matrix, nBestIndex)
                elif self.compare == "SELF-QE-HARD":
                    # print(f"sim_matrix:{sim_matrix}")
                    sim_matrix = self.softmax(sim_matrix)
                    # print(
                    #     f"sim_matrix after hard label softmax:{torch.sum(sim_matrix, dim = -1)}"
                    # )

                sim_value = torch.diagonal(sim_matrix, 0)

                if self.useTopOnly:
                    top_index = labels == 1
                    top_sim = sim_value[top_index]
                    if self.reduction == "sum":
                        contrastLoss = torch.neg(torch.sum(torch.log(top_sim)))
                    elif self.reduction == "mean":
                        contrastLoss = torch.neg(torch.mean(torch.log(top_sim)))
                else:
                    if self.reduction == "sum":
                        contrastLoss = torch.neg(torch.sum(torch.log(sim_value)))
                    elif self.reduction == "mean":
                        contrastLoss = torch.neg(torch.mean(torch.log(sim_value)))

            elif self.compare in ["SELF-LSTM", "SELF-LSTM-HARD"]:
                bert_embedding = output.last_hidden_state
                lstm_state, (_, _) = self.LSTM(bert_embedding)

                avg_pool = self.avgPool(input_ids, lstm_state, attention_mask)
                max_pool = self.maxPool(input_ids, lstm_state, attention_mask)

                concat_state = torch.cat([avg_pool, max_pool], dim=-1)
                proj_state = self.concatLSTM(concat_state)

                sim_matrix = torch.tensordot(bert_output, proj_state, dims=([1], [1]))
                cls_norm = torch.norm(bert_output, dim=-1)
                proj_norm = torch.norm(proj_state, dim=-1)
                sim_matrix = (sim_matrix / proj_norm) / cls_norm.unsqueeze(-1)
                assert (
                    torch.abs(torch.diagonal(sim_matrix, 0)) > 1.0
                ).count_nonzero() == 0, (
                    f"doing wrong cos sim:{torch.abs(torch.diagonal(sim_matrix, 0))}"
                )
                sim_matrix = sim_matrix / self.temperature
                if self.compare == "SELF-LSTM":
                    sim_matrix = self.activation_fn(sim_matrix, nBestIndex)
                elif self.compare == "SELF-LSTM-HARD":
                    sim_matrix = self.softmax(sim_matrix)
                sim_value = torch.diagonal(sim_matrix, 0)

                # print(f"sim_value:{sim_value.shape}")

                if self.useTopOnly:
                    top_index = labels == 1
                    top_sim = sim_value[top_index]
                    if self.reduction == "sum":
                        contrastLoss = torch.neg(torch.sum(torch.log(top_sim)))
                    elif self.reduction == "mean":
                        contrastLoss = torch.neg(torch.mean(torch.log(top_sim)))
                else:
                    if self.reduction == "sum":
                        contrastLoss = torch.neg(torch.sum(torch.log(sim_value)))
                    elif self.reduction == "mean":
                        contrastLoss = torch.neg(torch.mean(torch.log(sim_value)))

            elif self.compare in ["SELF-CSE", "SELF-CSE-HARD"]:
                dropout_1 = output.last_hidden_state[:, 0]
                dropout_2 = self.bert(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state[
                    :, 0
                ]  # forward again

                norm_1 = torch.norm(dropout_1, dim=-1)[:, None]
                norm_2 = torch.norm(dropout_2, dim=-1)[:, None]

                # print(f"dropout:{dropout_1.shape}")
                # print(f"norm:{norm_1.shape}")

                dropout_1_norm = dropout_1 / torch.max(
                    norm_1, 1e-6 * torch.ones(norm_1.shape, device=dropout_1.device)
                )
                dropout_2_norm = dropout_2 / torch.max(
                    norm_1, 1e-6 * torch.ones(norm_2.shape, device=dropout_2.device)
                )

                sim_matrix = torch.matmul(
                    dropout_1_norm, dropout_2_norm.transpose(0, 1)
                )
                # sim_matrix = (sim_matrix / norm_1) / norm_2

                sim_matrix = sim_matrix / self.temperature
                if self.compare == "SELF-CSE":
                    sim_matrix = self.activation_fn(sim_matrix, nBestIndex)
                elif self.compare == "SELF-CSE-HARD":
                    sim_matrix = self.softmax(sim_matrix)
                sim_value = torch.diagonal(sim_matrix, 0)
                if self.useTopOnly:
                    top_index = labels == 1
                    top_sim = sim_value[top_index]
                    if self.reduction == "sum":
                        contrastLoss = torch.neg(torch.sum(torch.log(top_sim)))
                    if self.reduction == "mean":
                        contrastLoss = torch.neg(torch.mean(torch.log(top_sim)))
                else:
                    if self.reduction == "sum":
                        contrastLoss = torch.neg(torch.sum(torch.log(sim_value)))
                    if self.reduction == "mean":
                        contrastLoss = torch.neg(torch.mean(torch.log(sim_value)))

            elif self.compare == "POOL":
                bert_embedding = output.last_hidden_state
                pooled_embedding = self.pooling(
                    input_ids, bert_embedding, attention_mask
                )
                sim_matrix = self.cosSim(bert_output, pooled_embedding)
                sim_matrix = self.activation_fn(sim_matrix, nBestIndex)
                if self.useTopOnly:
                    top_index = labels == 1
                    top_sim = sim_value[top_index]
                    contrastLoss = torch.neg(torch.sum(torch.log(top_sim)))
                else:
                    sim_value = torch.diag(sim_matrix, 0)
                    contrastLoss = torch.neg(torch.sum(sim_value))

            elif self.compare in ["REF", "REF-HARD"]:
                with torch.no_grad():
                    ref_output = self.bert(
                        input_ids=ref_ids, attention_mask=ref_mask
                    ).pooler_output
                    ref_output = self.dropout(ref_output)
                sim_matrix = torch.tensordot(ref_output, bert_output, dims=([1], [1]))

                ref_norm = torch.norm(ref_output, dim=-1)
                sim_matrix = sim_matrix / ref_norm.unsqueeze(-1)
                cls_norm = torch.norm(bert_output, dim=-1)
                sim_matrix = sim_matrix / cls_norm

                if self.compare == "REF-HARD":
                    sim_matrix = self.softmax(sim_matrix)
                else:
                    sim_matrix = self.activation_fn(sim_matrix, nBestIndex)
                top_index = (labels == 1).nonzero()

                contrastLoss = torch.tensor([0.0]).cuda()
                for i, batch_top in enumerate(top_index):
                    contrastLoss += torch.log(sim_matrix[i][batch_top])
                contrastLoss = torch.neg(contrastLoss)
                if self.reduction == "mean":
                    contrastLoss = contrastLoss / ref_output.shape[0]

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

    def pretrain(
        self,
        input_ids,
        attention_mask,
        labels,
        classifier_labels,
        add_classify=True,
        *args,
        **kwargs,
    ):
        """Only used in layer_op == LSTM currently"""
        assert "LSTM" in self.compare, "Only used in layer_op == LSTM currently"

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = output.last_hidden_state

        lstm_output, (h, c) = self.LSTM(bert_output)

        lm_output = self.LSTMLMHead(lstm_output)
        LM_loss = self.CE(lm_output.transpose(1, 2), labels)

        if add_classify:
            max_pool = self.maxPool(input_ids, lstm_output, attention_mask)
            avg_pool = self.avgPool(input_ids, lstm_output, attention_mask)

            concat_state = torch.cat([max_pool, avg_pool], dim=-1)

            scores = self.pretrain_classifier(concat_state).squeeze(-1)
            # print(f"scores:{scores}")
            # print(f"classifier:{classifier_labels}")
            classifier_loss = self.BCE(scores, classifier_labels)

            pretrain_loss = LM_loss + classifier_loss
        else:
            pretrain_loss = LM_loss
            classifier_loss = None

        return {
            "loss": pretrain_loss,
            "LM_loss": LM_loss,
            "classifier_loss": classifier_loss,
        }

    def parameters(self):
        parameters = list(self.bert.parameters()) + list(self.linear.parameters())

        if "LSTM" in self.compare:
            parameters = (
                parameters
                + list(self.LSTM.parameters())
                + list(self.concatLSTM.parameters())
                + list(self.LSTMLMHead.parameters())
            )

        return parameters

    def state_dict(self):
        checkpoint = {
            "bert": self.bert.state_dict(),
            "linear": self.linear.state_dict(),
        }
        if "LSTM" in self.compare:
            checkpoint["LSTM"] = self.LSTM.state_dict()
            checkpoint["concatLSTM"] = self.concatLSTM.state_dict()
            # checkpoint["concatCLSLinear"] = self.concatCLSLinear.state_dict()
            checkpoint["LSTMLMHead"] = self.LSTMLMHead.state_dict()

        return checkpoint

    def load_state_dict(self, checkpoint):
        self.bert.load_state_dict(checkpoint["bert"])
        self.linear.load_state_dict(checkpoint["linear"])

        # if self.layerOp is not None and self.layerOp == "LSTM":
        #     self.LSTM.load_state_dict(checkpoint["LSTM"])
        #     self.concatLSTM.load_state_dict(checkpoint["concatLSTM"])
        #     self.concatCLSLinear.load_state_dict(checkpoint["concatCLSLinear"])

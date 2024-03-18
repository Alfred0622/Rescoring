from collections import OrderedDict
import sys

from torch import Tensor

sys.path.append("../")
import numpy as np
import torch
import logging
from torch.nn.functional import log_softmax
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertConfig,
    AutoModelForCausalLM,
    BertLayer,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from src_utils.getPretrainName import getBertPretrainName
from utils.Pooling import MaxPooling, AvgPooling, MinPooling
from transformers import BertModel, BertConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer, KLDivLoss
from utils.activation_function import SoftmaxOverNBest
import torch.nn as nn


def check_sublist(src, target):
    for element in src:
        if not (element in target):
            return False

    return True


class NBestAttentionLayer(BertModel):
    def __init__(self, input_dim, device, n_layers):
        self.input_dim = input_dim
        self.hidden_size = input_dim
        self.num_hidden_layers = n_layers
        config = BertConfig(
            hidden_size=self.input_dim,
            num_hidden_layers=n_layers,
            attention_dropout=0.1,
        )
        BertModel.__init__(self, config)

    def forward(self, inputs_embeds, attention_matrix):
        return_dict = self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify inputs_embeds")

        batch_size, seq_length = input_shape
        device = inputs_embeds.device

        attention_matrix = attention_matrix.unsqueeze(0)

        # print(f'attention_mask:{attention_matrix.shape}')
        # print(f'attention_matrix:{attention_matrix}')
        # print(f'input_embeds.shape:{inputs_embeds.shape}')
        # print(f'check:{check.shape}')
        # print(f'attention_matrix:{attention_matrix.shape}')

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_matrix, input_shape
        )

        # print(f"extended_attention_mask:{extended_attention_mask.shape}")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # print(f"inputs_embeds:{inputs_embeds}")

        embedding_output = self.embeddings(
            input_ids=None,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )

        # print(f'embedding_output.shape:{embedding_output.shape}')
        # print(f"embedding_output:{embedding_output}")
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # print(f"attention_weight:{encoder_outputs.attentions[-1].shape}")
        attention_weight = encoder_outputs.attentions[-1].sum(dim=1)
        check = attention_weight != 0
        # print(f'check:{check.shape}')
        # print(f'attention_matrix:{attention_matrix.shape}')
        assert torch.not_equal(check, attention_matrix.bool()).count_nonzero() == 0

        # print(f"attention_weight:{encoder_outputs.attentions[-1]}")

        # print(f'pooler_output:{pooled_output}')

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class nBestCrossBert(torch.nn.Module):
    def __init__(
        self,
        dataset,
        device,
        lstm_dim=1024,
        use_fuseAttention=False,
        use_learnAttnWeight=False,
        addRes=False,
        fuseType="None",
        lossType="KL",
        concatCLS=False,
        wers = None,
        avg_wers = None,
        dropout=0.1,
        sepTask=False,
        taskType="WER",
        useRank=False,
        noCLS=True,
        noSEP=False,
    ):
        print(f"noCLS:{noCLS}, noSEP:{noSEP}")
        super().__init__()
        pretrain_name = getBertPretrainName(dataset)

        self.activation_fn = SoftmaxOverNBest()

        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.fuseType = fuseType
        self.lossType = lossType
        self.KL = lossType == "KL"
        self.useRank = useRank
        self.taskType = taskType
        self.noCLS = noCLS
        self.noSEP = noSEP
        self.resCLS = addRes

        print(f"taskType:{self.taskType}")

        self.maxPool = MaxPooling(self.noCLS, self.noSEP)
        self.minPool = MinPooling(self.noCLS, self.noSEP)
        self.avgPool = AvgPooling(self.noCLS, self.noSEP)

        self.l2Loss = torch.nn.MSELoss()

        self.use_fuseAttention = use_fuseAttention
        self.use_learnAttnWeight = use_learnAttnWeight

        if self.fuseType == "query":
            self.clsWeight = 0.5
            self.maskWeight = 0.5

        if not concatCLS:
            print(f"NO concatCLS")
            assert fuseType in [
                "lstm",
                "attn",
                "query",
                "sep_query",
                "none",
            ], "fuseType must be 'lstm', 'attn', or 'none'"
        else:
            print(f"Using concatCLS")
            assert fuseType in [
                "lstm",
                "attn",
                "query",
                "sep_query",
            ], f"fuseType must be 'lstm', 'attn' when concatCLS = True, but got fuseType = {fuseType}, concatCLS = {concatCLS}"

        # Model setting
        self.BCE = torch.nn.BCELoss()
        self.bert = BertModel.from_pretrained(pretrain_name)

        if fuseType == "lstm":
            self.lstm = torch.nn.LSTM(
                input_size=768,
                hidden_size=1024,
                num_layers=2,
                dropout=0.1,
                batch_first=True,
                bidirectional=True,
                proj_size=768,
            )

            self.concat_dim = 2 * 768 if concatCLS else 2 * 2 * 768

        elif fuseType == "attn":
            encoder_layer = TransformerEncoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=3072,
                layer_norm_eps=1e-12,
                batch_first=True,
                device=device,
            )

            self.attnLayer = TransformerEncoder(encoder_layer, num_layers=1)

            self.concat_dim = 2 * 768

        elif fuseType in ["attn", "none", "query"]:  # none
            self.concat_dim = (
                2 * 768 if not concatCLS else 768
            )  # if concatCLS, we only use the CLS part before and after LSTM / Attention

        self.concatLinear = torch.nn.Sequential(
            nn.Dropout(0.1), torch.nn.Linear(self.concat_dim, 768), torch.nn.ReLU()
        )  # concat the pooling output
        self.clsConcatLinear = torch.nn.Sequential(
            torch.nn.Linear(2 * 768, 768), torch.nn.ReLU()
        )  # concat the fused pooling output with CLS
        self.finalLinear = torch.nn.Sequential(torch.nn.Linear(770, 1))

        # NBest Attention Related
        if use_fuseAttention:
            self.clsConcatLinear = torch.nn.Linear(2 * 768, 766)
            self.fuseAttention = NBestAttentionLayer(
                input_dim=768, device=device, n_layers=1
            ).to(device)
            # print(f"NBestAttention Layer:{self.fuseAttention}")

            self.finalLinear = torch.nn.Linear(768, 1)
        else:
            self.fuseAttention = None

    def forward(
        self,
        input_ids,
        attention_mask,
        # crossAttentionMask,
        am_score,
        ctc_score,
        # ref_ids,
        # ref_masks,
        labels=None,
        nBestIndex=None,
        use_cls_loss=False,
        use_mask_loss=False,
        wers=None,
        *args,
        **kwargs,
    ):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls = output.pooler_output  # (B, 768)

        token_embeddings = output.last_hidden_state
        if self.resCLS:  # concat the CLS vector before and after the fusion
            if self.fuseType in ["lstm", "attn"]:
                if self.fuseType == "lstm":
                    fuse_state, (h, c) = self.lstm(token_embeddings)[:, 0, :]

                elif self.fuseType == "attn":
                    attn_mask = attention_mask.clone()

                    non_mask = attn_mask == 1
                    mask_index = attn_mask == 0

                    attn_mask[non_mask] = 0
                    attn_mask[mask_index] = 1

                    fuse_state = self.attnLayer(
                        src=token_embeddings, src_key_padding_mask=attn_mask
                    )[:, 0, :]

            elif self.fuseType == "query":
                mask_index = (input_ids == 103).nonzero(as_tuple=False).transpose(1, 0)
                fuse_state = token_embeddings[mask_index[0], mask_index[1], :]

            concatTrans = self.concatLinear(fuse_state).squeeze(1)

        else:  # notConcatCLS
            if self.fuseType in ["lstm", "attn", "none"]:
                if self.fuseType == "lstm":
                    fuse_state, (h, c) = self.lstm(token_embeddings)  # (B, L, 768)
                    attn_mask = attention_mask.clone()

                elif self.fuseType == "attn":
                    attn_mask = attention_mask.clone()

                    non_mask = attn_mask == 1
                    mask_index = attn_mask == 0

                    attn_mask[non_mask] = 0
                    attn_mask[mask_index] = 1

                    fuse_state = self.attnLayer(
                        src=token_embeddings, src_key_padding_mask=attn_mask
                    )
                elif self.fuseType == "none":
                    fuse_state = token_embeddings
                    attn_mask = attention_mask.clone()

                avg_state = self.avgPool(input_ids, fuse_state, attn_mask)
                max_state = self.maxPool(input_ids, fuse_state, attn_mask)
                concat_state = torch.cat([avg_state, max_state], dim=-1)

                concatTrans = self.concatLinear(concat_state).squeeze(1)

                # if self.sepTask:
                #     cls_concat = torch.cat([cls, am_score], dim=-1)
                #     cls_concat = torch.cat([cls_concat, ctc_score], dim=-1)

                #     mask_concat = torch.cat([concatTrans, am_score], dim=-1)
                #     mask_concat = torch.cat([mask_concat, ctc_score], dim=-1)
                #     cls_score = self.finalLinear(cls_concat).squeeze(-1)
                #     mask_score = self.finalExLinear(mask_concat).squeeze(-1)

                #     cls_prob = self.activation_fn(cls_score, nBestIndex)

                #     if self.taskType == "GT":
                #         cls_loss = labels * torch.log(cls_prob)
                #         cls_loss = torch.sum(torch.neg(cls_loss)).float()

                #         ref_cls = self.bert(
                #             input_ids=ref_ids, attention_mask=ref_masks
                #         ).pooler_output

                #         oracle_index = labels == 1
                #         oracle_cls = cls[oracle_index].clone()

                #         mask_loss = self.l2Loss(oracle_cls, ref_cls)

                #         loss = cls_loss + mask_loss

                #         logits = cls_score

                #     elif self.taskType == "WER":
                #         cls_loss = labels * torch.log(cls_prob)
                #         cls_loss = torch.sum(torch.neg(cls_loss))

                #         mask_loss = self.l2Loss(wers, mask_score)

                #         loss = cls_loss + mask_loss

                #         if not self.useRank:
                #             logits = cls_score - mask_score
                #     else:
                #         logits = cls_score

                # attMap = None

                # return {
                #     "loss": loss,
                #     "score": logits,
                #     "attention_map": attMap,
                #     "cls_loss": cls_loss,
                #     "mask_loss": mask_loss,
                # }

            elif self.fuseType == "query":
                mask_index = (input_ids == 103).nonzero(as_tuple=False).transpose(1, 0)
                mask_embedding = token_embeddings[mask_index[0], mask_index[1], :]

                cls_concat = torch.cat([cls, am_score], dim=-1)
                cls_concat = torch.cat([cls_concat, ctc_score], dim=-1)

                mask_concat = torch.cat([mask_embedding, am_score], dim=-1)
                mask_concat = torch.cat([mask_concat, ctc_score], dim=-1)

                cls_score = self.finalLinear(cls_concat).squeeze(-1)
                mask_score = self.finalExLinear(mask_concat).squeeze(-1)

                loss = None
                cls_loss = None
                mask_loss = None

                if not self.training:
                    if wers is None:
                        logits = (1 - self.clsWeight) * cls_score + (
                            1 - self.maskWeight
                        ) * mask_score
                    else:
                        if self.useRank:
                            start_index = 0
                            cls_rank = torch.zeros(cls_score.shape)
                            mask_rank = torch.zeros(mask_score.shape)
                            for index in nBestIndex:
                                (
                                    _,
                                    cls_rank[start_index : start_index + index],
                                ) = torch.sort(
                                    cls_score[
                                        start_index : start_index + index
                                    ].clone(),
                                    dim=-1,
                                    descending=True,
                                    stable=True,
                                )

                                (
                                    _,
                                    mask_rank[start_index : start_index + index],
                                ) = torch.sort(
                                    mask_score[
                                        start_index : start_index + index
                                    ].clone(),
                                    dim=-1,
                                    stable=True,
                                )
                                start_index += index

                            cls_rank = torch.reciprocal(1 + cls_rank)
                            mask_rank = torch.reciprocal(1 + mask_rank)

                            logits = cls_rank + mask_rank

                        else:
                            if self.useRank:
                                start_index = 0
                                cls_rank = torch.zeros(cls_score.shape)
                                mask_rank = torch.zeros(mask_score.shape)
                                for index in nBestIndex:
                                    (
                                        _,
                                        cls_rank[start_index : start_index + index],
                                    ) = torch.sort(
                                        cls_score[
                                            start_index : start_index + index
                                        ].clone(),
                                        dim=-1,
                                        descending=True,
                                        stable=True,
                                    )
                                    (
                                        _,
                                        mask_rank[start_index : start_index + index],
                                    ) = torch.sort(
                                        mask_score[
                                            start_index : start_index + index
                                        ].clone(),
                                        dim=-1,
                                        stable=True,
                                    )
                                    start_index += index

                                cls_rank = torch.reciprocal(1 + cls_rank)
                                mask_rank = torch.reciprocal(1 + mask_rank)

                                logits = cls_rank + mask_rank

                            else:
                                logits = cls_score - mask_score
                else:
                    logits = None

                if labels is not None:
                    start_index = 0

                    cls_score = self.activation_fn(
                        cls_score, nBestIndex, log_score=False
                    )  # softmax Over NBest

                    cls_loss = torch.log(cls_score) * labels
                    cls_loss = torch.neg(torch.mean(cls_loss))

                    if wers is not None:
                        mask_loss = self.l2Loss(mask_score, wers)
                    else:
                        mask_loss = labels * torch.log(mask_score)
                        mask_loss = torch.neg(torch.mean(mask_loss))

                    if use_cls_loss:
                        if use_mask_loss:
                            loss = cls_loss + mask_loss
                        else:
                            loss = cls_loss
                    else:
                        if use_mask_loss:
                            loss = mask_loss
                        else:
                            raise ValueError(
                                "there must be be at least one value set to True in use_cls_loss and get_mask_loss when fuseType == 'query' and concatCLS = False"
                            )

                attMap = None
                return {
                    "loss": loss,
                    "score": logits,
                    "attention_map": attMap,
                    "cls_loss": cls_loss,
                    "mask_loss": mask_loss,
                }

        cls_concat_state = torch.cat([cls, concatTrans], dim=-1)

        cls_concat_state = self.dropout(cls_concat_state)
        clsConcatTrans = self.clsConcatLinear(cls_concat_state)
        clsConcatTrans = self.dropout(clsConcatTrans)

        clsConcatTrans = torch.cat([clsConcatTrans, am_score], dim=-1)
        clsConcatTrans = torch.cat([clsConcatTrans, ctc_score], dim=-1).float()

        # NBest Attention
        attMap = None
        if self.fuseAttention is not None:
            clsConcatTrans = clsConcatTrans.unsqueeze(
                0
            )  # (B, D) -> (1, B ,D) here we take B as length
            output = self.fuseAttention(
                inputs_embeds=clsConcatTrans, attention_matrix=crossAttentionMask
            )
            scoreAtt = output.last_hidden_state
            attMap = output.attentions

            if self.addRes:
                scoreAtt = clsConcatTrans + scoreAtt

            finalScore = self.finalLinear(scoreAtt)
        # get final score
        else:
            finalScore = self.finalLinear(clsConcatTrans)

        # calculate Loss
        finalScore = finalScore.squeeze(-1)
        if self.fuseAttention:
            finalScore = finalScore.squeeze(0)
        logits = finalScore.clone()

        loss = None
        if labels is not None:
            assert nBestIndex is not None, "Must have N-best Index"
            start_index = 0
            for index in nBestIndex:
                finalScore[start_index : start_index + index] = self.activation_fn(
                    finalScore[start_index : start_index + index].clone(), nBestIndex
                )
                start_index += index

            if self.lossType == "KL":
                loss = self.loss(finalScore, labels)
            elif self.lossType == "BCE":
                loss = self.BCE(finalScore, labels)
            else:
                loss = labels * torch.log(finalScore)
                loss = torch.neg(loss)
                loss = torch.mean(loss)

        return {"loss": loss, "score": logits, "attention_map": attMap}

    def recognize(
            self, 
            input_ids,
            attention_mask,
            # crossAttentionMask,
            am_score,
            ctc_score,
            *args,
            **kwargs
        ):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.pooler_output  # (B, 768)
        token_embeddings = output.last_hidden_state
        fuse_state, (h, c) = self.lstm(token_embeddings)  # (B, L, 768)
        attn_mask = attention_mask.clone()

        avg_state = self.avgPool(input_ids, fuse_state, attn_mask)
        max_state = self.maxPool(input_ids, fuse_state, attn_mask)
        concat_state = torch.cat([avg_state, max_state], dim=-1)

        concatTrans = self.concatLinear(concat_state).squeeze(1)

        cls = torch.concat([cls, concatTrans], dim = -1)
        cls = self.clsConcatLinear(cls)
        cls = torch.concat([cls, am_score], dim = -1)
        cls = torch.concat([cls, ctc_score], dim = -1)

        logits = self.finalLinear(cls).squeeze(-1)

        return {
            "score": logits
        }


    def parameters(self):
        parameter = (
            list(self.bert.parameters())
            + list(  # list(self.concatLinear.parameters()) + \
                self.clsConcatLinear.parameters()
            )
            + list(self.finalLinear.parameters())
        )
        if self.use_fuseAttention:
            parameter = parameter + list(self.fuseAttention.parameters())

        if hasattr(self, "lstm"):
            parameter += list(self.lstm.parameters())
        if hasattr(self, "attnLayer"):
            parameter += list(self.attnLayer.parameters())
        if hasattr(self, "concatLinear"):
            parameter += list(self.concatLinear.parameters())
        if hasattr(self, "finalExLinear"):
            parameter += list(self.finalExLinear.parameters())

        return parameter
    
    def show_param(self):
        print(sum(p.numel() for p in self.parameters()))

    def state_dict(self):
        fuseAttend = None
        if self.use_fuseAttention:
            fuseAttend = self.fuseAttention.state_dict()

        state_dict = {
            "bert": self.bert.state_dict(),
            # "concatLinear":  self.concatLinear.state_dict(),
            "clsConcatLinear": self.clsConcatLinear.state_dict(),
            "finalLinear": self.finalLinear.state_dict(),
            "fuseAttend": fuseAttend,
        }

        if hasattr(self, "lstm"):
            state_dict["lstm"] = self.lstm.state_dict()

        if hasattr(self, "attnLayer"):
            state_dict["attnLayer"] = self.attnLayer.state_dict()

        if hasattr(self, "concatLinear"):
            state_dict["concatLinear"] = self.concatLinear.state_dict()

        if hasattr(self, "clsWeight"):
            state_dict["clsWeight"] = self.clsWeight

        if hasattr(self, "maskWeight"):
            state_dict["maskWeight"] = self.maskWeight

        if hasattr(self, "finalExLinear"):
            state_dict["finalExLinear"] = self.finalExLinear.state_dict()

        state_dict["noCLS"] = self.noCLS
        state_dict["noSEP"] = self.noSEP

        return state_dict

    def load_state_dict(self, checkpoint):
        self.bert.load_state_dict(checkpoint["bert"])
        if hasattr(self, "lstm"):
            self.lstm.load_state_dict(checkpoint["lstm"])
        if hasattr(self, "attnLayer"):
            self.attnLayer.load_state_dict(checkpoint["attnLayer"])

        if hasattr(self, "concatLinear") and "concatLinear" in checkpoint.keys():
            self.concatLinear.load_state_dict(checkpoint["concatLinear"])

        self.clsConcatLinear.load_state_dict(checkpoint["clsConcatLinear"])
        self.finalLinear.load_state_dict(checkpoint["finalLinear"])
        if checkpoint["fuseAttend"] is not None:
            self.fuseAttention.load_state_dict(checkpoint["fuseAttend"])
        if hasattr(self, "finalExLinear") and "finalExLinear" in checkpoint.keys():
            self.finalExLinear.load_state_dict(checkpoint["finalExLinear"])

        if "noCLS" in checkpoint.keys():
            self.noCLS = checkpoint["noCLS"]
        if "noSEP" in checkpoint.keys():
            self.noSEP = checkpoint["noSEP"]

    def set_weight(self, cls_loss, mask_loss, is_weight=False):
        if is_weight:
            self.clsWeight = cls_loss
            self.maskWeight = mask_loss
        else:
            print(f"[cls_loss, mask_loss]:{torch.tensor([cls_loss, mask_loss]).shape}")
            print(f"[cls_loss, mask_loss]:{torch.tensor([cls_loss, mask_loss])}")
            softmax_loss = torch.softmax(torch.tensor([cls_loss, mask_loss]), dim=-1)
            print(f"softmax_loss:{softmax_loss}")
            self.clsWeight = softmax_loss[0]
            self.maskWeight = softmax_loss[1]

        print(f"\n clsWeight:{self.clsWeight}\n maskWeight:{self.maskWeight}")


class pBertSimp(torch.nn.Module):
    def __init__(
        self,
        args,
        train_args,
        device,
        output_attention=True,
    ):
        super().__init__()
        pretrain_name = getBertPretrainName(args["dataset"])
        config = BertConfig.from_pretrained(pretrain_name)
        config.output_attentions = True
        config.output_hidden_states = True
        self.combineScore = train_args['combineScore']
        self.addLMScore = train_args['addLMScore']

        if (self.combineScore and self.addLMScore):
            logging.warning("combineScore and addLMScore should not be True at same time, Linear dim will be 769")

        self.bert = BertModel(config=config).from_pretrained(pretrain_name).to(device)
        if (self.combineScore):
            self.linear = torch.nn.Linear(769, 1).to(device)
        elif (self.addLMScore):
            self.linear = torch.nn.Linear(771, 1).to(device)
        else:
            self.linear = torch.nn.Linear(770, 1).to(device)

        self.hardLabel = True
        self.loss_type = "Entropy"
        self.loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)

        self.output_attention = output_attention

        self.activation_fn = SoftmaxOverNBest()
        self.weightByWER = False

        self.reduction = train_args['reduction']

        print(f"output_attention:{self.output_attention}")
        print(f"weightByWER:{self.weightByWER}")

    def forward(
        self,
        input_ids,
        attention_mask,
        nBestIndex,
        am_score=None,
        ctc_score=None,
        lm_score = None,
        scores = None,
        labels=None,
        wers=None,  # not None if weightByWER = True
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

        if (self.combineScore):
            scores = scores.unsqueeze(-1)
            clsConcatoutput = torch.cat([output, scores], dim = -1)
        else:
            clsConcatoutput = torch.cat([output, am_score], dim=-1)
            clsConcatoutput = torch.cat([clsConcatoutput, ctc_score], dim=-1)
            if (self.addLMScore):
                lm_score = lm_score.unsqueeze(-1)
                clsConcatoutput = torch.cat([clsConcatoutput, lm_score], dim=-1)


        scores = self.linear(clsConcatoutput).squeeze(-1)
        # print(f'scores:{scores}')
        final_score = scores.clone().detach()

        loss = None
        if labels is not None:
            # print(f'labels:{labels}')
            scores = self.activation_fn(scores, nBestIndex, log_score=False)
            loss = labels * torch.log(scores)
            if (self.reduction == 'mean'):
                loss = torch.mean(torch.neg(loss))
            elif (self.reduction == 'sum'):
                loss = torch.sum(torch.neg(loss))
        
        # print(f'final_scores:{final_score}')

        return {
            "loss": loss,
            "score": final_score,
            "attention_weight": bert_output.attentions,
        }
    def recognize(
        self,
        input_ids,
        attention_mask,
        am_score=None,
        ctc_score=None,
        *args,
        **kwargs):
        
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attention,
        ).pooler_output

        clsConcatoutput = torch.cat([output, am_score], dim=-1)
        clsConcatoutput = torch.cat([clsConcatoutput, ctc_score], dim=-1)
        scores = self.linear(clsConcatoutput).squeeze(-1)

        return {
            "score": scores
        }

    def parameters(self):
        parameters = list(self.bert.parameters()) + list(self.linear.parameters())
        return parameters

    def state_dict(self):
        return {
            "bert": self.bert.state_dict(),
            "linear": self.linear.state_dict(),
        }
    def load_state_dict(self, checkpoint):
        self.bert.load_state_dict(checkpoint['bert'])
        self.linear.load_state_dict(checkpoint['linear'])
    
    def show_param(self):
        # print(f'parameters:{self.parameters()}')
        print(sum(p.numel() for p in self.parameters()))

class pBert(torch.nn.Module):
    def __init__(
        self,
        args,
        train_args,
        device,
        output_attention=True,
    ):
        super().__init__()
        pretrain_name = getBertPretrainName(args["dataset"])
        config = BertConfig.from_pretrained(pretrain_name)
        config.output_attentions = True
        config.output_hidden_states = True

        self.bert = BertModel(config=config).from_pretrained(pretrain_name).to(device)
        self.linear = torch.nn.Linear(770, 1).to(device)

        self.hardLabel = train_args["hard_label"]
        self.loss_type = train_args["loss_type"]
        self.loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.BCE = torch.nn.BCELoss()

        self.dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)

        self.output_attention = output_attention

        self.reduction = train_args["reduction"]

        self.activation_fn = SoftmaxOverNBest()
        self.weightByWER = train_args["weightByWER"]
        self.layer_op = train_args["layer_op"]
        self.MWER = train_args["MWER"]

        extra_layer_config = BertConfig.from_pretrained(pretrain_name)
        extra_layer_config.num_hidden_layers = 1

        if self.layer_op is not None and "lastInit" in self.layer_op:
            init_layer = self.layer_op.split("_")[-1]
            print(f"last init:{init_layer}")
            for i in range(1, int(init_layer) + 1):
                print(f"layer:-{i}")
                self.bert.encoder.layer[-i] = BertLayer(config)
        elif self.layer_op == "extra":
            self.extra_layer = BertLayer(config)
        elif self.layer_op == "LSTM":
            self.LSTM = torch.nn.LSTM(
                input_size=768,
                hidden_size=1024,
                batch_first=True,
                num_layers=1,
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

        print(f"output_attention:{self.output_attention}")
        print(f"weightByWER:{self.weightByWER}")

    def forward(
        self,
        input_ids,
        attention_mask,
        nBestIndex,
        am_score=None,
        ctc_score=None,
        labels=None,
        wers=None,  # not None if weightByWER = True
        avg_wers = None,
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

        clsConcatoutput = torch.cat([output, am_score], dim=-1)
        clsConcatoutput = torch.cat([clsConcatoutput, ctc_score], dim=-1)

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

            # print(f"loss after entropy:{loss}")

            if wers is not None:
                if self.weightByWER is not None:
                    if self.weightByWER == "inverse":  # Lower WER get larger Weight
                        wers = torch.reciprocal(1 + 10 * wers)
                    elif self.weightByWER == "positive":  # Higher WER get higher weight
                        wers = 0.5 + (wers * 5)
                    elif self.weightByWER == "square":  # WER
                        wers = ((wers - 0.2) + 1) ** 2  #
                    loss = loss * wers

                if self.MWER is not None and self.MWER == "MWED":
                    start_index = 0
                    temperature = torch.zeros(
                        softmax_wers.shape, device=final_score.device
                    )
                    for N in nBestIndex:
                        temp_score = torch.sum(
                            final_score[start_index : start_index + N]
                        )
                        temp_wer = torch.sum(wers[start_index : start_index + N])
                        temperature[start_index : start_index + N] = (
                            temp_score / temp_wer
                        )
                        smoothed_score = final_score / temperature
                        smoothed_score = self.activation_fn(smoothed_score, nBestIndex)
                        start_index += N

                    discriminative_loss = torch.neg(
                        softmax_wers * torch.log(smoothed_score)
                    )
                    # print(f"discriminative_loss:{discriminative_loss}")
                    loss = loss + discriminative_loss
    
                elif (self.MWER == 'MWER'):
                    print(f'wers:{wers.shape}')
                    print(f'avg_wers:{avg_wers.shape}')
                    sub_wers = wers - avg_wers
                    softmax_wers = self.activation_fn(wers, nBestIndex)
                    discriminative_loss = scores * sub_wers
                    # print(f"discriminative_loss:{discriminative_loss}")
                    loss = loss + discriminative_loss
            else:
                if self.reduction == "sum":
                    loss = torch.sum(loss)
                elif self.reduction == "mean":
                    loss = torch.mean(loss)
        
        # print(f'final_scores:{final_score}')
        return {
            "loss": loss,
            "score": final_score,
            "attention_weight": bert_output.attentions,
        }

    def recognize_by_attention(
        self,
        input_ids,
        attention_mask,
        # am_score,
        # ctc_score,
        # nBestIndex,
    ):
        bert_output = self.bert(input_ids, attention_mask, output_attentions=True)
        attention_weight_heads = bert_output.attentions  # [B, 1, L , 12]
        # print(f"attention_weight_heads:{len(attention_weight_heads)}")
        last_attention_weight = attention_weight_heads[-1]
        # print(f"attention_weight_heads[0]:{last_attention_weight.shape}")
        attention_weight = last_attention_weight.sum(dim=1) / 12
        # print(f"attention_weight:{attention_weight.shape}")
        # print(f"attention_mask:{attention_mask.shape}")
        length = (
            attention_mask.sum(dim=1).unsqueeze(-1).expand(-1, attention_mask.shape[1])
        )
        target = 1 / length
        cls_attention_weight = attention_weight[:, 0, :]
        # print(f'check:{torch.sum(attention_weight, dim = -1)}')
        target[attention_mask == 0] = 0
        target[input_ids == 102] = 1
        target[input_ids == 101] = 0

        cls_attention_weight[attention_mask == 0] = 1e-9
        cls_attention_weight[input_ids == 101] = 1e-9

        # print(f"target:{target}")

        entropy = torch.neg(
            torch.sum(target * (torch.log(cls_attention_weight)), dim=-1)
        )

        # print(f"score:{entropy}")

        return {"score": entropy, "loss": None}

    def parameters(self):
        parameters = list(self.bert.parameters()) + list(self.linear.parameters())

        if self.layer_op == "extra":
            parameters += list(self.extra_layer.parameters())

        return parameters

    def state_dict(self):
        state_dict = {
            "bert": self.bert.state_dict(),
            "linear": self.linear.state_dict(),
        }

        if self.layer_op == "extra":
            state_dict["extra_layer"] = self.extra_layer.state_dict()

        return state_dict

    def show_param(self):
        print(sum(p.numel() for p in self.parameters()))

    # def load_state_dict(self, state_dict):
    # self.bert.load_state_dict(state_dict["bert"])
    # self.linear.load_state_dict(state_dict["linear"])

    # if self.layer_op == "extra":
    #     self.extra_layer.load_state_dict(state_dict["extra_layer"])


class nBestfuseBert(torch.nn.Module):
    def __init__(self, args, train_args, device, **kwargs):
        super().__init__()
        pretrain_name = getBertPretrainName(args["dataset"])
        self.nBest = args["nbest"]
        self.bert = BertModel.from_pretrained(pretrain_name)
        self.fuse_config = BertConfig.from_pretrained(pretrain_name)
        self.fuse_config.num_hidden_layers = 1

        self.fuse_model = NBestAttentionLayer(input_dim=768, device=device, n_layers=2)
        self.finalLinear = torch.nn.Linear(770, 1)

        self.activation_fn = SoftmaxOverNBest()

        # print(f'bert:{self.bert.config}')
        # print(f"fuse:{self.fuse_config}")

    def forward(
        self,
        input_ids,
        attention_mask,
        nBestMask,
        am_score,
        ctc_score,
        nBestIndex,
        labels=None,
        *args,
        **kwargs,
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        batch_size = input_ids.shape[0] // self.nBest

        cls = output.pooler_output.unsqueeze(0)

        # cls = cls.view(batch_size, self.nBest, -1)
        nBestFuseCLS = self.fuse_model(
            inputs_embeds=cls, attention_matrix=nBestMask
        ).last_hidden_state.squeeze(0)

        nBestFuseCLS = nBestFuseCLS + cls
        
        nBestFuseCLS = nBestFuseCLS.squeeze(0)

        # am_score = am_score.view(batch_size, -1, 1)
        # ctc_score = ctc_score.view(batch_size, -1, 1)
        nBestFuseCLS = torch.cat([nBestFuseCLS, am_score], dim=-1)
        nBestFuseCLS = torch.cat([nBestFuseCLS, ctc_score], dim=-1)

        nBestScore = self.finalLinear(nBestFuseCLS).squeeze(-1)

        nBestProb = self.activation_fn(nBestScore.clone(), nBestIndex, topk=50)

        # print(f"nBestProb:{nBestProb}")

        if labels is not None:
            loss = labels * torch.log(nBestProb)
            loss = torch.neg(torch.mean(loss))

        # print(f"label:{labels}")
        # print(f"loss:{loss}")

        # print(f"score:{nBestScore}")
        # print(f"prob:{nBestProb}")

        return {"score": nBestScore, "loss": loss}

    def parameters(self):
        return (
            list(self.bert.parameters())
            + list(self.fuse_model.parameters())
            + list(self.finalLinear.parameters())
        )
    def show_param(self):
        print(sum(p.numel() for p in self.paramters()))

    def state_dict(self):
        return {
            "bert": self.bert.state_dict(),
            "fuse_model": self.fuse_model.state_dict(),
            "finalLinear": self.finalLinear.state_dict(),
        }

    def load_state_dict(self, checkpoint):
        self.bert.load_state_dict(checkpoint["bert"])
        self.fuse_model.load_state_dict(checkpoint["fuse_model"])
        self.finalLinear.load_state_dict(checkpoint["finalLinear"])


class inter_pBert(torch.nn.Module):
    def __init__(
        self,
        args,
        train_args,
        device,
        output_attention=True,
    ):
        super().__init__()
        pretrain_name = getBertPretrainName(args["dataset"])
        config = BertConfig.from_pretrained(pretrain_name)
        config.output_attentions = True
        config.output_hidden_states = True

        self.bert = BertModel(config=config).from_pretrained(pretrain_name).to(device)
        self.linear = torch.nn.Linear(770, 1).to(device)

        self.hardLabel = True
        self.loss_type = "Entropy"
        self.loss = torch.nn.KLDivLoss(reduction="batchmean")

        self.output_attention = output_attention

        self.activation_fn = SoftmaxOverNBest()
        self.weightByWER = False

        print(f"output_attention:{self.output_attention}")
        print(f"weightByWER:{self.weightByWER}")

    def forward(
        self,
        input_ids,
        attention_mask,
        nBestIndex,
        am_score=None,
        ctc_score=None,
        labels=None,
        wers=None,  # not None if weightByWER = True
        *args,
        **kwargs,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attention,
        )

        output = bert_output.pooler_output

        clsConcatoutput = torch.cat([output, am_score], dim=-1)
        clsConcatoutput = torch.cat([clsConcatoutput, ctc_score], dim=-1)

        scores = self.linear(clsConcatoutput).squeeze(-1)
        final_score = scores.clone().detach()

        loss = None
        if labels is not None:
            if self.hardLabel:
                scores = self.activation_fn(scores, nBestIndex, log_score=False)
                loss = labels * torch.log(scores)
                loss = torch.neg(loss)
            else:
                if self.loss_type == "KL":
                    scores = self.activation_fn(
                        scores, nBestIndex, log_score=True
                    )  # Log_Softmax
                    loss = self.loss(scores, labels)
                else:
                    scores = self.activation_fn(scores, nBestIndex, log_score=False)
                    loss = (
                        torch.sum(labels * torch.log(scores)) / input_ids.shape[0]
                    )  # batch_mean
                    loss = torch.neg(loss)

            if wers is not None:
                if self.weightByWER == "inverse":  # Lower WER get larger Weight
                    wers = torch.reciprocal(1 + 10 * wers)
                elif self.weightByWER == "positive":  # Higher WER get higher weight
                    wers = 0.5 + (wers * 5)
                elif self.weightByWER == "square":  # WER
                    wers = ((wers - 0.2) + 1) ** 2  #
                loss = loss * wers
                loss = torch.sum(loss) / input_ids.shape[0]  # batch_mean

        return {
            "loss": loss,
            "score": final_score,
            "attention_weight": bert_output.attentions,
        }


class poolingBert(torch.nn.Module):
    def __init__(self, args, train_args, **kwargs):
        super().__init__()

        pretrain_name = getBertPretrainName(args["dataset"])

        self.bert = BertModel.from_pretrained(pretrain_name)

        self.minPooling = MinPooling(train_args["noCLS"], train_args["noSEP"])
        self.maxPooling = MaxPooling(train_args["noCLS"], train_args["noSEP"])
        self.avgPooling = AvgPooling(train_args["noCLS"], train_args["noSEP"])

        self.pooling_type = train_args["pooling_type"].split()

        assert (
            1 <= len(self.pooling_type) and len(self.pooling_type) <= 3
        ), "1 ~ 3 Pooling method"
        assert check_sublist(
            self.pooling_type, ["min", "max", "avg"]
        ), "only may have 'min', 'max' or 'mean' "

        self.pooling_type = self.pooling_type
        self.concat_linear = torch.nn.Sequential(
            torch.nn.Linear(768 * len(self.pooling_type), 768), torch.nn.ReLU()
        )
        self.cls_concat_linear = torch.nn.Sequential(
            torch.nn.Linear(768 * 2, 768), torch.nn.ReLU()
        )
        self.final_linear = torch.nn.Linear(770, 1)

        self.activation_fn = SoftmaxOverNBest()

    def forward(
        self,
        input_ids,
        attention_mask,
        am_score,
        ctc_score,
        nBestIndex,
        labels=None,
        wers=None,
        *args,
        **kwargs,
    ):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls = bert_output.pooler_output

        token_embeddings = bert_output.last_hidden_state
        if "min" in self.pooling_type:
            pooling_vector = self.minPooling(token_embeddings, attention_mask)
            if "max" in self.pooling_type:
                pooling_vector = torch.cat(
                    [pooling_vector, self.maxPooling(token_embeddings, attention_mask)],
                    dim=-1,
                )
            if "avg" in self.pooling_type:
                pooling_vector = torch.cat(
                    [pooling_vector, self.avgPooling(token_embeddings, attention_mask)],
                    dim=-1,
                )
        elif "max" in self.pooling_type:
            pooling_vector = self.maxPooling(token_embeddings, attention_mask)
            if "avg" in self.pooling_type:
                pooling_vector = torch.cat(
                    [pooling_vector, self.avgPooling(token_embeddings, attention_mask)],
                    dim=-1,
                )
        elif "avg" in self.pooling_type:
            pooling_vector = self.avgPooling(token_embeddings, attention_mask)
        # (768 * number of pooling -> 768)

        project_vector = self.concat_linear(pooling_vector)

        # concatenate with CLS
        cls_concat = torch.cat([cls, project_vector], dim=-1)
        project_vector = self.cls_concat_linear(cls_concat)

        score_concat = torch.cat([project_vector, am_score], dim=-1)
        score_concat = torch.cat([score_concat, ctc_score], dim=-1)

        final_score = self.final_linear(score_concat).squeeze(1)
        final_prob = self.activation_fn(final_score.clone(), nBestIndex)

        if labels is not None:
            loss = labels * torch.log(final_prob)
            loss = torch.neg(torch.sum(loss))

            assert not torch.isnan(loss), f"loss = Nan"
        else:
            loss = None

        return {"score": final_score, "loss": loss}

    def parameters(self):
        parameters = (
            list(self.bert.parameters())
            + list(self.concat_linear.parameters())
            + list(self.final_linear.parameters())
        )

        return parameters

    def state_dict(self):
        return {
            "bert": self.bert.state_dict(),
            "pool_concat": self.concat_linear.state_dict(),
            "cls_concat": self.cls_concat_linear.state_dict(),
            "final": self.final_linear.state_dict(),
        }

    def load_state_dict(self, checkpoint):
        self.bert.load_state_dict(checkpoint["bert"])
        self.concat_linear.load_state_dict(checkpoint["pool_concat"])
        self.cls_concat_linear.load_state_dict(checkpoint["cls_concat"])
        self.final_linear.load_state_dict(checkpoint["final"])

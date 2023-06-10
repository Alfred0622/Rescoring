import sys

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


class NBestAttentionLayer(BertModel):
    def __init__(self, input_dim, device, n_layers):
        self.input_dim = input_dim
        self.hidden_size = input_dim
        self.num_hidden_layers = n_layers
        config = BertConfig(
            dim=self.input_dim, num_hidden_layers=n_layers, attention_dropout=0.3
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

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_matrix, input_shape)

        # print(f"extended_attention_mask:{attention_matrix.shape}")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]

        embedding_output = self.embeddings(
            input_ids=None,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_matrix,
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
        lstm_dim=512,
        use_fuseAttention=False,
        use_learnAttnWeight=False,
        addRes=False,
        fuseType="None",
        lossType="KL",
        concatCLS=False,
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

        self.activation_func = SoftmaxOverNBest()

        self.addRes = addRes
        self.resCLS = concatCLS
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.fuseType = fuseType
        self.KL = lossType == "KL"
        self.useRank = useRank
        self.sepTask = sepTask
        self.taskType = taskType
        self.noCLS = noCLS
        self.noSEP = noSEP

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
        self.bert = BertModel.from_pretrained(pretrain_name)

        if fuseType == "lstm":
            self.lstm = torch.nn.LSTM(
                input_size=768,
                hidden_size=lstm_dim,
                num_layers=2,
                dropout=0.1,
                batch_first=True,
                bidirectional=True,
            )

            self.concat_dim = 2 * lstm_dim if concatCLS else 2 * 2 * lstm_dim

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
            torch.nn.Linear(self.concat_dim, 768), torch.nn.ReLU()
        )  # concat the pooling output
        self.clsConcatLinear = torch.nn.Sequential(
            torch.nn.Linear(2 * 768, 768), torch.nn.ReLU()
        )  # concat the fused pooling output with CLS
        self.finalLinear = torch.nn.Linear(770, 1)

        if self.sepTask:
            self.finalExLinear = torch.nn.Linear(770, 1)

        # NBest Attention Related
        if use_fuseAttention:
            self.clsConcatLinear = torch.nn.Linear(2 * 768, 766)
            self.fuseAttention = NBestAttentionLayer(
                input_dim=768, device=device, n_layers=1
            ).to(device)
            print(f"NBestAttention Layer:{self.fuseAttention}")

            self.finalLinear = torch.nn.Linear(768, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        crossAttentionMask,
        am_score,
        ctc_score,
        ref_ids,
        ref_masks,
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
            concatTrans = self.dropout(concatTrans)

        else:  # notConcatCLS
            if self.fuseType in ["lstm", "attn", "none"]:
                if self.fuseType == "lstm":
                    fuse_state, (h, c) = self.lstm(token_embeddings)  # (B, L, 768)

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

                avg_state = self.avgPool(fuse_state, attn_mask)
                max_state = self.maxPool(fuse_state, attn_mask)
                concat_state = torch.cat([avg_state, max_state], dim=-1)
                concatTrans = self.concatLinear(concat_state).squeeze(1)
                concatTrans = self.dropout(concatTrans)

                if self.sepTask:
                    cls_concat = torch.cat([cls, am_score], dim=-1)
                    cls_concat = torch.cat([cls_concat, ctc_score], dim=-1)

                    mask_concat = torch.cat([concatTrans, am_score], dim=-1)
                    mask_concat = torch.cat([mask_concat, ctc_score], dim=-1)
                    cls_score = self.finalLinear(cls_concat).squeeze(-1)
                    mask_score = self.finalExLinear(mask_concat).squeeze(-1)

                    cls_prob = self.activation_func(cls_score, nBestIndex)

                    if self.taskType == "GT":
                        cls_loss = labels * torch.log(cls_prob)
                        cls_loss = torch.sum(torch.neg(cls_loss)).float()

                        ref_cls = self.bert(
                            input_ids=ref_ids, attention_mask=ref_masks
                        ).pooler_output

                        oracle_index = labels == 1
                        oracle_cls = cls[oracle_index].clone()

                        mask_loss = self.l2Loss(oracle_cls, ref_cls)

                        loss = cls_loss + mask_loss

                        logits = cls_score

                    elif self.taskType == "WER":
                        cls_loss = labels * torch.log(cls_prob)
                        cls_loss = torch.sum(torch.neg(cls_loss))

                        mask_loss = self.l2Loss(wers, mask_score)

                        loss = cls_loss + mask_loss

                        if not self.useRank:
                            logits = cls_score - mask_score
                    else:
                        logits = cls_score

                    attMap = None

                    return {
                        "loss": loss,
                        "score": logits,
                        "attention_map": attMap,
                        "cls_loss": cls_loss,
                        "mask_loss": mask_loss,
                    }

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

                    cls_score = self.activation_func(
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
                    finalScore[start_index : start_index + index].clone()
                )
                start_index += index

            if self.KL:
                loss = self.loss(finalScore, labels)
            else:
                loss = labels * torch.log(finalScore)
                loss = torch.neg(loss)

        return {"loss": loss, "score": logits, "attention_map": attMap}

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


class pBert(torch.nn.Module):
    def __init__(
        self,
        dataset,
        device,
        hardLabel=False,
        output_attention=True,
        loss_type="KL",
        weightByWER="postive",
    ):
        super().__init__()
        pretrain_name = getBertPretrainName(dataset)
        config = BertConfig.from_pretrained(pretrain_name)
        config.output_attentions = True
        config.output_hidden_states = True

        self.bert = BertModel(config=config).from_pretrained(pretrain_name).to(device)
        self.linear = torch.nn.Linear(770, 1).to(device)

        self.hardLabel = hardLabel
        self.loss_type = loss_type
        self.loss = torch.nn.KLDivLoss(reduction="batchmean")

        self.output_attention = output_attention

        self.activation_fn = SoftmaxOverNBest()
        self.weightByWER = weightByWER

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
                    wers = 1 / ((wers - 0.2) + 1) ** 2  #
                loss = loss * wers
                loss = torch.sum(loss) / input_ids.shape[0]  # batch_mean

        return {
            "loss": loss,
            "score": final_score,
            "attention_weight": bert_output.attentions,
        }

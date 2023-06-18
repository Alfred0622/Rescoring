import torch
import logging
from torch.nn.functional import log_softmax
from torch.optim import AdamW
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertConfig,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# from sentence_transformers import SentenceTransformer
from torch.nn.functional import log_softmax
from utils.cal_score import get_sentence_score
from torch.nn import AvgPool1d, MaxPool1d


class MLMBert(torch.nn.Module):
    def __init__(
        self,
        train_batch,
        test_batch,
        nBest,
        device,
        lr=1e-5,
        mode="random",
        pretrain_name="bert-base-chinese",
    ):
        torch.nn.Module.__init__(self)
        self.device = device
        self.model = BertForMaskedLM.from_pretrained(pretrain_name).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_name)
        self.mask = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.num_nBest = nBest
        self.mode = mode

        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def forward(self, input_id, mask):
        # random mask reference : https://github.com/jamescalam/transformers/blob/main/course/training/03_mlm_training.ipynb

        if self.mode == "random":
            label = input_id.clone().detach()
            rand = torch.rand(input_id.shape).to(self.device)
            mask_index = (
                (rand < 0.15) * (input_id != 101) * (input_id != 102) * (input != 0)
            )
            selection = []
            for i in range(input_id.shape[0]):
                selection.append(torch.flatten(mask_index[i].nonzero()).tolist())

            for i in range(input_id.shape[0]):
                input_id[i, selection[i]] = self.mask

            label[input_id != self.mask] = -100

            output = self.model(input_ids=input_id, attention_mask=mask, labels=label)

            loss = output.loss

            return loss

        elif self.mode == "sequence":
            batch_size = input_id.shape[0]

            mlm_seq = []
            mlm_mask = []
            labels = []
            for i in range(batch_size):
                id_list = input_id[i].tolist()
                sep = id_list.index(102)
                mask_seq = input_id[i].clone()
                att_mask = mask[i].clone()
                for j in range(1, sep):

                    selection = mask_seq[j].item()
                    mask_seq[j] = self.mask
                    mlm_seq.append(mask_seq.clone())
                    mlm_mask.append(att_mask)

                    label = mask_seq.clone()
                    label[label != self.mask] = -100
                    label[label == self.mask] = selection
                    labels.append(label)

                    mask_seq[j] = selection
            mlm_seq = torch.stack(mlm_seq).to(self.device)
            mlm_mask = torch.stack(mlm_mask).to(self.device)
            labels = torch.stack(labels).to(self.device)

            output = self.model(
                input_ids=mlm_seq, attention_mask=mlm_mask, labels=labels
            )

            loss = output.loss

            return loss

    def recognize(
        self, input_id, attention_mask, free_memory=True, sentence_per_process=40
    ):
        # generate PLL loss from teacher
        # only support batch = 1
        pll_score = []  # (batch_size, N-Best)
        pll_input = []
        pll_mask = []
        mask_index = []
        seq_for_hyp = []

        no_token = set()

        if free_memory:
            tmps = []
            masks = []
            mask_index = []
            expand_num = 0
            pll_scores = []  # for return

            for j in range(input_id.shape[0]):  # for every hyp in this batch
                token_len = torch.sum(input_id[j] != 0, dim=-1)
                if token_len == 2:
                    pll_scores.append(10000)
                    continue

                mask = attention_mask[j].unsqueeze(0)
                tmp = input_id[j].clone().to(self.device)

                for k in range(1, token_len - 1):
                    to_mask = tmp[k].item()
                    tmp[k] = self.mask
                    tmps.append(tmp.clone())
                    tmp[k] = to_mask

                    masks.append(mask)

                    mask_index.append([expand_num, k, to_mask])
                    expand_num += 1

                seq_for_hyp.append(expand_num - 1)

                tmps = torch.stack(tmps)
                masks = torch.stack(masks)
                mask_index = torch.tensor(mask_index)
                # Mask index will be like below here:
                # [ [0,1,v1], [0,2,v2], [0,3,v3]...etc]

                outputs = self.model(input_ids=tmps, attention_mask=masks)
                outputs = log_softmax(outputs[0], dim=-1)
                # logging.warning(f'mask index{mask_index}')
                mask_index = torch.transpose(mask_index, 0, 1)
                # logging.warning(f'mask index after transpose:{mask_index}')
                # [ [0,1,v1], [0,2,v2], [0,3,v3]...etc ]
                # -> [
                # [0,0,0 ... etc], [1,2,3,...etc], [v1,v2,v3... etc]
                # ]
                pll_score = outputs[mask_index[0], mask_index[1], mask_index[2]]
                # logging.warning(f'pll_score:{pll_score}')

                accum_score = 0.0
                for i, score in enumerate(pll_score):
                    accum_score += score
                # logging.warning(f'accum_score:{accum_score}')
                pll_scores.append(accum_score)
                accum_score = 0.0

                tmps = []
                masks = []
                mask_index = []
                seq_for_hyp = []
                expand_num = 0
            if len(pll_scores) != input_id.shape[0]:
                logging.warning(f"pll_scores:{len(pll_scores)}")
                logging.warning(f"input_id:{input_id.shape[0]}")
                logging.warning(f"pll_scores:{pll_scores}")
                logging.warning(f"input_id:{input_id}")
            assert (
                len(pll_scores) == input_id.shape[0]
            ), f"{len(pll_score)} != {input_id.shape[0]}"
            pll_scores = torch.tensor(pll_scores)
            # logging.warning(f'pll_scores:{pll_scores}')

            return pll_scores

            # for j in range(input_id.shape[0]):  # for every hyp in this batch
            #     # set 10-nbest each time
            #     token_len = torch.sum(input_id[j] != 0, dim=-1)
            #     logging.warning(f'{j} -- token len:{token_len}')
            #     if token_len == 2:
            #         pll_scores.append(-10000)
            #         continue

            #     mask = attention_mask[j].unsqueeze(0)
            #     tmp = input_id[j].clone().to(self.device)

            #     final_score = 0.0
            #     for k in range(1, token_len-1):
            #         to_mask = tmp[k].item()
            #         tmp[k] = self.mask
            #         tmps.append(tmp.clone())
            #         tmp[k] = to_mask

            #         masks.append(mask)

            #         mask_index.append([expand_num, k, to_mask]) #(num of row, index, char)
            #         expand_num += 1

            #         if (k % sentence_per_process == 0) or (k == (token_len - 2)):
            #             tmps = torch.stack(tmps)
            #             masks = torch.stack(masks)
            #             mask_index = torch.tensor(mask_index)

            #             outputs = self.model(input_ids = tmps, attention_mask = masks)
            #             outputs = log_softmax(outputs[0], dim=-1)
            #             mask_index = torch.transpose(mask_index, 0, 1)
            #             pll_score = outputs[mask_index[0], mask_index[1], mask_index[2]]

            #             logging.warning(f'{j} -- calculate score:{pll_score}')

            #             for i, score in enumerate(pll_score):
            #                 final_score += score

            #             tmps = []
            #             masks = []
            #             mask_index = []
            #             expand_num = 0

            #     logging.warning(f'{j} -- final_score:{final_score}')
            #     pll_scores.append(final_score)

            # pll_scores = torch.tensor(pll_scores)

            # return pll_scores

        else:
            expand_num = 0
            for j in range(input_id.shape[0]):  # for every hyp in this batch
                token_len = torch.sum(input_id[j] != 0, dim=-1)
                if token_len == 2:
                    no_token.add(j)
                    continue

                mask = attention_mask[j]
                tmp = input_id[j].clone()
                for k in range(
                    1, token_len - 1
                ):  # for each token in this hyp (exclude padding)
                    to_mask = tmp[k].item()
                    tmp[k] = self.mask

                    pll_input.append(tmp.clone())
                    pll_mask.append(mask)
                    mask_index.append([expand_num, k, to_mask])
                    expand_num += 1
                    tmp[k] = to_mask

                seq_for_hyp.append(expand_num - 1)

            pll_input = torch.stack(pll_input).to(self.device)
            pll_mask = torch.stack(pll_mask).to(self.device)
            mask_index = torch.tensor(mask_index)

            outputs = self.model(input_ids=pll_input, attention_mask=pll_mask)
            outputs = log_softmax(outputs[0], dim=-1)
            mask_index = torch.transpose(mask_index, 0, 1)
            pll_score = outputs[mask_index[0], mask_index[1], mask_index[2]].tolist()

            pll = []
            count = 0
            accum_score = 0.0
            for i, score in enumerate(pll_score):
                accum_score += score
                if i in seq_for_hyp:
                    pll.append(accum_score)
                    accum_score = 0.0
                    count += 1
                if count in no_token:
                    pll.append(-10000)
                    count += 1

        pll_score = torch.tensor(pll)

        return pll_score


class RescoreBert(torch.nn.Module):
    def __init__(
        self,
        train_batch,
        test_batch,
        nBest,
        device,
        mode="MD",
        weight=1.0,
        use_MWER=False,
        use_MWED=False,
        lr=1e-5,
        pretrain_name="bert-base-chinese",
    ):
        torch.nn.Module.__init__(self)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_name)
        # self.config = DistilBertConfig(vocab_size = self.tokenizer.vocab_size, n_layers=4)
        self.model = BertModel.from_pretrained(pretrain_name).to(device)
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.nBest = nBest
        self.weight = weight
        self.use_MWER = use_MWER
        self.use_MWED = use_MWED
        self.device = device
        self.l2_loss = torch.nn.MSELoss()
        self.mode = mode

        print(f"use_MWER:{use_MWER}, use_MWED:{use_MWED}")

        self.fc = torch.nn.Linear(768, 1).to(device)
        model_parameter = list(self.model.parameters()) + list(self.fc.parameters())
        # self.optimizer = AdamW(self.model.parameters(), lr = lr)
        self.optimizer = AdamW(model_parameter, lr=lr)

        # if self.mode == "SimCSE":
        #     self.model = SentenceTransformer("cyclone/simcse-chinese-roberta-wwm-ext")

    def forward(self, input_id, text, attention_mask, first_scores, cers, pll_score):
        """
        input_id : (B * N_Best, Seq)
        """

        total_loss = 0.0

        loss_MWER = None
        loss_MWED = None

        batch_size = input_id.shape[0]

        if self.mode == "SimCSE":
            embedding = torch.from_numpy(self.model.encode(text))

            score = self.fc(embedding).view(pll_score.shape)

            total_loss = self.l2_loss(score, pll_score)
        elif self.mode == "MD":
            s_output = self.model(input_ids=input_id, attention_mask=attention_mask)
            s_score = self.fc(s_output.last_hidden_state[:, 0, :])

            ignore_index = pll_score == -10000
            # logging.warning(f's_score before view:{s_score}')
            s_score = s_score.view(pll_score.shape)
            # logging.warning(f'pll_score.shape:{pll_score.shape}')

            distill_score = s_score.clone()
            distill_score[ignore_index] = -10000

            total_loss = self.l2_loss(distill_score, pll_score)
            weight_sum = first_scores + self.weight * s_score.view(first_scores.shape)

            # MWER
            if self.use_MWER:
                wer = torch.stack(
                    [((s[1] + s[2] + s[3]) / (s[0] + s[1] + s[2])) for s in cers]
                ).to(self.device)
                weight_sum = weight_sum.reshape(batch_size, -1)
                wer = wer.reshape(batch_size, -1)  # (B, N-best)

                p_hyp = torch.softmax(weight_sum, dim=-1)
                avg_error = torch.mean(wer, dim=-1).unsqueeze(-1)
                loss_MWER = p_hyp * (wer - avg_error)

                sub_wer = wer - avg_error

                loss_MWER = torch.sum(loss_MWER)

                total_loss = loss_MWER + 1e-4 * total_loss

            # MWED
            elif self.use_MWED:
                wer = torch.stack(
                    [((s[1] + s[2] + s[3]) / (s[0] + s[1] + s[2])) for s in cers]
                ).to(self.device)

                wer = wer.reshape(batch_size, -1)
                weight_sum = weight_sum.reshape(batch_size, -1)

                T = torch.sum(weight_sum, dim=-1) / torch.sum(
                    wer, dim=-1
                )  # hyperparameter T
                T = T.unsqueeze(-1)

                d_error = torch.softmax(wer, dim=-1)
                d_score = torch.softmax(weight_sum / T, dim=-1).reshape(d_error.shape)

                loss_MWED = d_error * torch.log(d_score)
                loss_MWED = torch.neg(torch.sum(loss_MWED))

                total_loss = loss_MWED + 1e-4 * total_loss

        if not self.training:
            weight_sum = weight_sum.view(self.test_batch, -1)
            cers = cers.view(self.test_batch, -1, 4)
            best_hyp = torch.argmax(weight_sum)
            return total_loss, cers[0][best_hyp]

        return total_loss

    def recognize(self, input_id, text, attention_mask, first_scores, weight=1.0):
        batch_size = input_id.shape[0]
        if self.mode == "SimCSE":
            embedding = torch.from_numpy(self.model.encode(text))
            rescore = self.fc(embedding).view(first_scores.shape)
        else:
            output = self.model(input_ids=input_id, attention_mask=attention_mask)
            rescore = self.fc(output[0][:, 0, :]).view(first_scores.shape)

        weighted_score = first_scores + (weight * rescore)
        weighted_score = weighted_score.view(batch_size, -1)

        max_sentence = torch.argmax(weighted_score, dim=-1)
        best_hyps = input_id[max_sentence].tolist()

        best_hyp_tokens = []
        for hyp in best_hyps:
            sep = hyp.index(102)
            best_hyp_tokens.append(self.tokenizer.convert_ids_to_tokens(hyp[1:sep]))

        return rescore, weighted_score, best_hyp_tokens, max_sentence


class RescoreBertAlsem(torch.nn.Module):
    def __init__(self, dataset, lstm_dim, device):
        super().__init__()

        bert_pretrain_name = "None"
        gpt_pretrain_name = "None"

        if dataset in ["aishell", "aishell2"]:
            bert_pretrain_name = "bert-base-chinese"
            gpt_pretrain_name = "ckiplab/gpt2-base-chinese"
        elif dataset in ["tedlium2", "librispeech"]:
            bert_pretrain_name = "bert-base-uncased"
            gpt_pretrain_name = "gpt2"
        elif dataset in ["csj"]:
            bert_pretrain_name = "cl-tohoku/bert-base-japanese"
            gpt_pretrain_name = "ClassCat/gpt2-base-japanese-v2"

        self.device = device

        self.bert = BertModel.from_pretrained(bert_pretrain_name).to(device)
        # self.gpt = AutoModelForCausalLM.from_pretrained(gpt_pretrain_name)

        self.biLSTM = torch.nn.LSTM(
            input_size=768,
            hidden_size=lstm_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=True,
        ).to(device)

        self.concatLinear = torch.nn.Linear(4 * lstm_dim, lstm_dim).to(device)
        self.scoreLinear = torch.nn.Sequential(
            torch.nn.Linear(
                768 + lstm_dim + 2, 768
            ),  # bert dim + lastm_dim + [am_score, ctc_score]
            torch.nn.ReLU(),
            torch.nn.Linear(768, 1),
        ).to(device)

        self.l2_loss = torch.nn.MSELoss()
        self.NllLoss = torch.nn.NLLLoss()

    def forward(
        self,
        bert_ids,
        bert_mask,
        gpt_scores,
        am_scores,
        ctc_scores,
    ):

        # gpt_states = self.gpt(gpt_ids, gpt_mask).logits

        # gpt_score = log_softmax(gpt_states, dim = -1)
        # gpt_score = get_sentence_score(gpt_score, gpt_ids, gpt_bos, gpt_eos, gpt_pad)

        cls = self.bert(bert_ids, bert_mask).pooler_output
        last_hidden_state = self.bert(bert_ids, bert_mask).last_hidden_state[:, :, :]

        LSTM_state, (h, c) = self.biLSTM(last_hidden_state)

        # print(f"LSTM:{LSTM_state.shape}")

        avg_pool = AvgPool1d(LSTM_state.shape[1]).to(self.device)
        max_pool = MaxPool1d(LSTM_state.shape[1]).to(self.device)

        # print(f"LSTM:{LSTM_state.shape}")

        avg_state = torch.transpose(avg_pool(torch.transpose(LSTM_state, 1, 2)), 1, 2)

        max_state = torch.transpose(max_pool(torch.transpose(LSTM_state, 1, 2)), 1, 2)

        concatStates = torch.cat([avg_state, max_state], dim=-1).squeeze(1)
        # print(concatStates.shape)
        projStates = self.concatLinear(concatStates)

        # print(f'cls:{cls.shape}')
        # print(f'projState:{projStates.shape}')

        concatCLS = torch.cat([cls, projStates], dim=-1)
        # print(f"concatCLS.shape:{concatCLS.shape}")
        # print(f"am_scores.shape:{am_scores.shape}")
        # print(f"ctc_scores.shape:{ctc_scores.shape}")
        scores = torch.cat(
            [am_scores.unsqueeze(-1), ctc_scores.unsqueeze(-1)], dim=-1
        ).to(cls.device)
        # print(scores.shape)
        concatCLS = torch.cat([concatCLS, scores], dim=-1)
        scores = self.scoreLinear(concatCLS).squeeze(-1)

        # print(f"score:{scores.shape}")
        # print(f"gpt_score:{gpt_scores.shape}")

        l2_loss = self.l2_loss(scores, gpt_scores)

        return SequenceClassifierOutput(
            loss=l2_loss,
            logits=scores,
        )

    def recognize(
        self,
        bert_ids,
        bert_mask,
        am_scores,
        ctc_scores,
    ):
        cls = self.bert(bert_ids, bert_mask).pooler_output
        last_hidden_state = self.bert(bert_ids, bert_mask).last_hidden_state

        LSTM_state, (h, c) = self.biLSTM(last_hidden_state)

        avg_pool = AvgPool1d(LSTM_state.shape[1]).to(self.device)
        max_pool = MaxPool1d(LSTM_state.shape[1]).to(self.device)

        avg_state = torch.transpose(avg_pool(torch.transpose(LSTM_state, 1, 2)), 1, 2)
        max_state = torch.transpose(max_pool(torch.transpose(LSTM_state, 1, 2)), 1, 2)

        concatStates = torch.cat([avg_state, max_state])
        projStates = self.concatLinear(concatStates)

        scores = torch.cat(
            [am_scores.unsqueeze(-1), ctc_scores.unsqueeze(-1)], dim=-1
        ).to(cls.device)
        concatCLS = torch.cat([cls, projStates])
        concatCLS = torch.cat([concatCLS, scores])
        scores = torch.scoreLinear(concatCLS)

        return scores

    def parameters(self):
        return (
            list(self.bert.parameters())
            + list(self.biLSTM.parameters())
            + list(self.concatLinear.parameters())
            + list(self.scoreLinear.parameters())
        )

    def checkpoint(self):
        return {
            "BERT": self.bert.state_dict(),
            "LSTM": self.biLSTM.state_dict(),
            "concatLinear": self.concatLinear.state_dict(),
            "scoreLinear": self.scoreLinear.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.bert.load_state_dict(state_dict["BERT"])
        self.biLSTM.load_state_dict(state_dict["LSTM"])
        self.concatLinear.load_state_dict(state_dict["concatLinear"])
        self.scoreLinear.load_state_dict(state_dict["scoreLinear"])

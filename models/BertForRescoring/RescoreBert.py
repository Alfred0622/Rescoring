import torch
import logging
from torch.nn.functional import log_softmax
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertConfig,
)
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer

class MLMBert(torch.nn.Module):
    def __init__(self, train_batch, test_batch, nBest, device, lr=1e-5, mode="random", pretrain_name = 'bert-base-chinese'):
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

    def recognize(self, input_id, attention_mask, free_memory = True):
        # generate PLL loss from teacher
        pll_score = []  # (batch_size, N-Best)
        pll_input = []
        pll_mask = []
        mask_index = []
        seq_for_hyp = []
        
        no_token = set()
        
        if (free_memory):
            pll_scores = []

            tmps = []
            masks = []
            mask_index = []
            expand_num = 0
            for j in range(input_id.shape[0]):  # for every hyp in this batch
                # set 10-nbest each time
                token_len = torch.sum(input_id[j] != 0, dim=-1)
                if token_len == 2:
                    pll_scores.append(10000)
                    continue

                mask = attention_mask[j].unsqueeze(0)
                tmp = input_id[j].clone().to(self.device)

                for k in range(1, token_len-1):
                    to_mask = tmp[k].item()
                    tmp[k] = self.mask
                    tmps.append(tmp.clone())
                    tmp[k] = to_mask

                    masks.append(mask)

                    mask_index.append([expand_num, k, to_mask])
                    expand_num += 1
                
                seq_for_hyp.append(expand_num - 1)
                
                if (j % 10 == 0) or (j == input_id.shape[0] - 1):
                    tmps = torch.stack(tmps)
                    masks = torch.stack(masks)
                    mask_index = torch.tensor(mask_index)
                
                    outputs = self.model(input_ids=tmps, attention_mask=masks)
                    outputs = log_softmax(outputs[0], dim=-1)
                    mask_index = torch.transpose(mask_index, 0, 1)
                    pll_score = outputs[mask_index[0], mask_index[1], mask_index[2]]
                    
                    accum_score = 0.0
                    for i, score in enumerate(pll_score):
                        accum_score += score
                        if i in seq_for_hyp:
                            pll_scores.append(accum_score)
                            accum_score = 0.0
                    tmps = []
                    masks = []
                    mask_index = []
                    seq_for_hyp = []
                    expand_num = 0
                else:
                    continue
            pll_scores = torch.tensor(pll_scores)
           
            return pll_scores

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
                    pll.append(10000)
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
    ):
        torch.nn.Module.__init__(self)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        # self.config = DistilBertConfig(vocab_size = self.tokenizer.vocab_size, n_layers=4)
        self.model = BertModel.from_pretrained('bert-base-chinese').to(device)
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.nBest = nBest
        self.weight = weight
        self.use_MWER = use_MWER
        self.use_MWED = use_MWED
        self.device = device
        self.l2_loss = torch.nn.MSELoss()
        self.mode = mode

        print(f'use_MWER:{use_MWER}, use_MWED:{use_MWED}')

        self.fc = torch.nn.Linear(768, 1).to(device)
        model_parameter = list(self.model.parameters()) + list(self.fc.parameters())
        # self.optimizer = AdamW(self.model.parameters(), lr = lr)
        self.optimizer = AdamW(model_parameter, lr=lr)

        if self.mode == "SimCSE":
            self.model = SentenceTransformer("cyclone/simcse-chinese-roberta-wwm-ext")

    def forward(self, input_id, text, attention_mask, first_scores, cers, pll_score):
        """
        input_id : (B * N_Best, Seq)
        """

        total_loss = 0.0

        loss_MWER = None
        loss_MWED = None

        if self.mode == "SimCSE":
            embedding = torch.from_numpy(self.model.encode(text))

            score = self.fc(embedding).view(pll_score.shape)

            total_loss = self.l2_loss(score, pll_score)
        elif self.mode == "MD":
            s_output = self.model(input_ids=input_id, attention_mask=attention_mask)
            s_score = self.fc(s_output[0][:, 0, :])

            ignore_index = pll_score == 10000
            # logging.warning(f's_score before view:{s_score}')
            s_score = s_score.view(pll_score.shape)

            distill_score = s_score.clone()
            distill_score[ignore_index] = 10000

            total_loss = self.l2_loss(distill_score, pll_score)
            weight_sum = first_scores + self.weight * s_score.view(first_scores.shape)

        # MWER
            if self.use_MWER:
                wer = torch.stack(
                    [((s[1] + s[2] + s[3]) / (s[0] + s[1] + s[2])) for s in cers]
                ).to(self.device)
                p_hyp = torch.softmax(weight_sum, dim=-1)
                avg_error = torch.mean(wer)
                avg_error = torch.full(wer.shape, avg_error).to(self.device)

                loss_MWER = p_hyp * (wer - avg_error)

                loss_MWER = loss_MWER.sum()

                total_loss = loss_MWER + 1e-4 * total_loss

        # MWED
            elif self.use_MWED:
                wer = torch.stack(
                    [((s[1] + s[2] + s[3]) / (s[0] + s[1] + s[2])) for s in cers]
                )
                d_error = torch.softmax(wer, dim=-1)
                d_score = torch.softmax(s_score, dim=-1)

                loss_MWED = d_error * torch.log(d_score.view(d_error.shape))
                loss_MWED = torch.neg(loss_MWED.sum())

                total_loss = loss_MWED + 1e-4 * total_loss

        if not self.training:
            weight_sum = weight_sum.view(self.test_batch, self.nBest)
            cers = cers.view(self.test_batch, self.nBest, -1)
            best_hyp = torch.argmax(weight_sum)

            return total_loss, cers[0][best_hyp]

        return total_loss

    def recognize(self, input_id, text, attention_mask, first_scores, weight=1.0):
        if self.mode == "SimCSE":
            embedding = torch.from_numpy(self.model.encode(text))
            rescore = self.fc(embedding).view(first_scores.shape)
        else:
            output = self.model(input_ids=input_id, attention_mask=attention_mask)
            rescore = self.fc(output[0][:, 0, :]).view(first_scores.shape)

        weighted_score = first_scores + (weight * rescore)
        weighted_score = weighted_score.view(self.test_batch, self.nBest)

        max_sentence = torch.argmax(weighted_score, dim=-1)
        best_hyps = input_id[max_sentence].tolist()

        best_hyp_tokens = []
        for hyp in best_hyps:
            sep = hyp.index(102)
            best_hyp_tokens.append(self.tokenizer.convert_ids_to_tokens(hyp[1:sep]))

        return rescore, weighted_score, best_hyp_tokens, max_sentence

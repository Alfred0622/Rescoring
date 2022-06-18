import logging
from espnet.nets import e2e_asr_common, asr_interface
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.asr.pytorch_backend.asr import load_trained_model
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
from torch.nn import TransformerDecoder as Decoder
from torch.nn.functional import log_softmax
from torch.optim import AdamW
from transformers import BertTokenizer
from models.transformer_utils.embedding import PositionalEmbedding


class AudioAwareReranker(nn.Module):
    def __init__(
        self,
        device,
        use_res=True,
        d_model=768,
        decoder_layers=6,
        use_spike=False,
        trigger_threshold=0.7,
        lr=1e-5,
        nbest=50,
    ):
        super().__init__()
        self.device = device

        asr, _ = load_trained_model(
            f"/mnt/nas3/Alfred/espnet/egs/aishell/asr1/exp/interctc/train_pytorch_20220320_12layers_6dec/results/model.last10.avg.best",
            training=False,
        )
        self.asr = asr.encoder.to(self.device)
        self.asr_ctc = asr.ctc.to(self.device)
        self.bert = BertModel.from_pretrained("bert-base-chinese").to(self.device)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, batch_first=True
        )
        self.odim = self.bert.config.vocab_size
        self.decoder = Decoder(
            decoder_layer,
            num_layers=decoder_layers,
        ).to(self.device)
        self.use_spike = use_spike
        self.trigger_threshold = trigger_threshold
        self.criterion = nn.CrossEntropyLoss()
        self.project = nn.Linear(256, 768).to(self.device)
        self.odim = self.bert.config.vocab_size
        self.fc = nn.Linear(768, self.odim).to(self.device)
        self.pe = PositionalEmbedding(d_model=d_model).to(self.device)

        self.use_res = use_res
        print(f"use_res:{use_res}")
        logging.warning(f"use_res:{use_res}")

        parameters = (
            list(self.decoder.parameters())
            + list(self.project.parameters())
            + list(self.fc.parameters())
        )
        self.optimizer = AdamW(parameters, lr=lr)
        self.nbest = nbest

        # load state dict of E2E
        for p in self.asr.parameters():
            p.requires_grad = False
        for p in self.asr_ctc.parameters():
            p.requires_grad = False
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(
        self, audio, ilens, input_ids, attention_mask, labels, use_pos_only=False
    ):
        xs_pad = audio[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        # forward asr encoder
        hs_pad, hs_mask, hs_intermediates = self.asr(xs_pad, src_mask)

        # calculate trigger threshold
        if self.use_spike:
            # forward CTC
            ctc_hid = self.asr_ctc.ctc_lo(hs_pad)  # (B, len, |V|)
            ctc_hid = log_softmax(ctc_hid, dim=-1)
            probs = torch.ones(ctc_hid.shape[:-1]).to(self.device)
            probs = probs - ctc_hid[:, :, 0]
            spike_index = (probs >= self.trigger_threshold).nonzero()

            logging.warning(f"spike_index:{spike_index}")

            # batchfy spikes
            spikes = []
            single_spike = []
            spike_mask = []
            single_spike_mask = []
            last_row = spike_index[0][0]

            logging.warning(f"spike_index:{spike_index}")

            # Append & stack spikes
            for index in spike_index:
                if index[0] != last_row:
                    last_row = index[0]
                    spikes.append(torch.stack(single_spike))
                    single_spike = []

                    spike_mask.append(torch.tensor(single_spike_mask))
                    single_spike_mask = []

                single_spike.append(hs_pad[index[0], index[1], :])
                single_spike_mask.append(False)

            spikes.append(torch.stack(single_spike))  # (B, Seq_len, 256)
            spike_mask.append(torch.tensor(single_spike_mask))

            # final spikes for cross attention
            hs_pad = pad_sequence(spikes, batch_first=True)
            hs_mask = pad_sequence(spike_mask, batch_first=True, padding_value=True)
            # hs_pad = hs_pad[spike_index[0], spike_index[1], :]

            logging.warning(f"hs_pad:{hs_pad.shape}")
            logging.warning(f"hs_mask:{hs_mask.shape}")
        else:
            hs_mask = torch.logical_not(hs_mask).squeeze(1)  #  Mask for Cross Attention

        tgt_mask = torch.logical_not(attention_mask)  #  Mask for Cross Attention

        # project asr hidden state from dim-256 to dim-768
        hs_pad = self.project(hs_pad)

        # token-level embedding
        if not use_pos_only:
            bert_embedding = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
        else:
            bert_embedding = torch.zeros(
                (labels.shape[0], labels.shape[1], 768).shape
            ).to(self.device)

        bert_embedding = self.pe(bert_embedding)
        tgt_attention_mask = torch.diag(
            torch.ones(bert_embedding.shape[1], dtype=torch.bool)
        ).to(self.device)

        # cross attention with asr encoder hidden state & token-level embedding
        output = self.decoder(
            bert_embedding,
            hs_pad,
            tgt_mask=tgt_attention_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=hs_mask,
        )

        if self.use_res:
            output = output + bert_embedding

        # proj to vocab-size & softmax
        output = self.fc(output)
        output = log_softmax(output, dim=-1).permute(0, 2, 1)

        loss = self.criterion(output, labels)

        return loss

    def recognize(self, audio, ilens, input_ids, attention_mask, use_pos_only=False):
        xs_pad = audio[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask, hs_intermediates = self.asr(xs_pad, src_mask)

        hs_mask = torch.logical_not(hs_mask)

        # calculate trigger threshold
        if self.use_spike:
            # forward CTC
            ctc_hid = self.asr_ctc.ctc_lo(hs_pad)  # (B, len, |V|)
            ctc_hid = log_softmax(ctc_hid, dim=-1)
            probs = torch.ones(ctc_hid.shape[:-1]).to(self.device)
            probs = probs - ctc_hid[:, :, 0]
            spike_index = (probs >= self.trigger_threshold).nonzero()

            # batchfy spikes
            spikes = []
            single_spike = []
            spike_mask = []
            single_spike_mask = []
            last_row = spike_index[0][0]

            # Append & stack spikes
            for i, j in spike_index:
                if i != last_row:
                    last_row = i
                    spikes.append(torch.stack(single_spike))
                    single_spike = []

                    spike_mask.append(torch.tensor(single_spike_mask))
                    single_spike_mask = []

                single_spike.append(hs_pad[i, j, :])
                single_spike_mask.append(False)

            spikes.append(torch.stack(single_spike))  # (B, Seq_len, 256)
            spike_mask.append(torch.tensor(single_spike_mask))

            # final spikes for cross attention
            hs_pad = pad_sequence(spikes, batch_first=True)
            hs_mask = pad_sequence(spike_mask, batch_first=True, padding_value=True)
            # hs_pad = hs_pad[spike_index[0], spike_index[1], :]
        else:
            hs_mask = torch.logical_not(hs_mask).squeeze(1)  #  Mask for Cross Attention

        tgt_mask = torch.logical_not(attention_mask)  #  Mask for Cross Attention

        hs_pad = self.project(hs_pad)

        if not use_pos_only:
            bert_embedding = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        else:
            bert_embedding = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], 768)
            ).to(self.device)

        bert_embedding = self.pe(bert_embedding)
        tgt_attention_mask = torch.diag(
            torch.ones(bert_embedding.shape[1], dtype=torch.bool)
        ).to(self.device)

        output = self.decoder(
            bert_embedding,
            hs_pad,
            tgt_mask=tgt_attention_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=hs_mask,
        )
        if self.use_res:
            output = output + bert_embedding

        logging.warning(f"output:{output}")

        scores = self.fc(output)
        logging.warning(f"scores before softmax:{scores}")

        scores = log_softmax(scores, dim=-1)

        logging.warning(f"scores:{scores}")

        total_score = []
        for i, token in enumerate(input_ids):
            score = 0.0
            len = 0
            for j, t in enumerate(token):
                if t == 101:
                    continue
                if t == 102:
                    total_score.append(score / len)
                    break
                score += scores[i][j][t]
                logging.warning(f"score:{score}")
                len += 1
        logging.warning(f"total_score:{total_score}")
        return torch.stack(total_score)  # (S_num)

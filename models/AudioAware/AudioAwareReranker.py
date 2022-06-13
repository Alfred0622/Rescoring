import logging
from espnet.nets import e2e_asr_common, asr_interface
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.asr.pytorch_backend.asr import load_trained_model
import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn import TransformerDecoder as Decoder
from torch.nn.functional import softmax
from torch.optim import AdamW
from transformers import BertTokenizer


class AudioAwareReranker(nn.Module):
    def __init__(
        self,
        device,
        d_model=768,
        decoder_layers=2,
        use_spike=True,
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

        self.fc = nn.Linear(768, self.odim).to(self.device)

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
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, audio, ilens, input_ids, attention_mask, labels):
        xs_pad = audio[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask, hs_intermediates = self.asr(audio, src_mask)

        # calculate trigger threshold
        if self.use_spike:
            ctc_hid = self.asr_ctc.ctc_lo(hs_pad)  # (B, len, |V|)
            ctc_hid = softmax(ctc_hid, dim=-1)
            probs = torch.ones(ctc_hid.shape[:-1]).to(self.device)
            probs = probs - ctc_hid[:, :, 0]
            spike_index = (probs >= self.trigger_threshold).nonzero()
            spike_index = torch.transpose(spike_index, 0, 1)
            # debug: 每個batch的spike數未必一樣
            hs_pad = hs_pad[spike_index[0], spike_index[1], :]

        hs_pad = self.project(hs_pad)
        bert_embedding = self.bert(input_ids=input_ids, attention_mask=attention_mask)[
            0
        ]

        output = self.decoder(bert_embedding, hs_pad)

        output = softmax(self.fc(output), dim=-1).permute(0, 2, 1)

        loss = self.criterion(output, labels)

        return loss

    def recognize(self, audio, ilens, input_ids, attention_mask):
        xs_pad = audio[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask, hs_intermediates = self.asr(xs_pad, src_mask)

        # calculate trigger threshold
        if self.use_spike:
            ctc_hid = self.asr_ctc.ctc_lo(hs_pad)  # (B, len, |V|)
            ctc_hid = softmax(ctc_hid, dim=-1)
            probs = torch.ones(ctc_hid.shape[:-1])
            probs = probs - ctc_hid[:, :, 0]
            spike_index = (probs > self.trigger_threshold).nonzero()[0]
            hs_pad = hs_pad[spike_index]

        hs_pad = self.project(hs_pad)

        bert_embedding = self.bert(input_ids=input_ids, attention_mask=attention_mask)[
            0
        ]

        logging.warning(bert_embedding.shape)
        logging.warning(hs_pad.shape)

        output = self.decoder(bert_embedding, hs_pad)
        scores = self.fc(output)

        scores = softmax(scores, dim=-1)

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
                len += 1
        return torch.stack(total_score)  # (S_num)

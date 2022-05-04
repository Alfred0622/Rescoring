from numpy import place
import torch
import torch.nn as nn
from torch.optim import AdamW
import logging
from transformers import (
    BertModel,
    BertTokenizer,
    BertGenerationEncoder,
    BertGenerationDecoder,
    BertGenerationTokenizer,
    EncoderDecoderModel,
    BertConfig,
    EncoderDecoderConfig,
)
from torch.nn.utils.rnn import pad_sequence
from nBestAligner.nBestAlign import align, alignNbest


class nBestTransformer(nn.Module):
    def __init__(
        self,
        nBest,
        train_batch,
        test_batch,
        device,
        lr=1e-5,
        mode="align",
        pretrain_name="bert-base-chinese",
    ):
        nn.Module.__init__(self)

        encoder = BertGenerationEncoder.from_pretrained(
            "bert-base-chinese", bos_token_id=101, eos_token_id=102
        )

        decoder = BertGenerationDecoder.from_pretrained(
            "bert-base-chinese",
            add_cross_attention=True,
            is_decoder=True,
            bos_token_id=101,
            eos_token_id=102,
        )

        self.device = device

        self.nBest = nBest
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.mode = mode

        self.tokenizer = BertTokenizer.from_pretrained(pretrain_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad = self.tokenizer.convert_tokens_to_ids("[PAD]")

        self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to(
            self.device
        )

        self.embedding = nn.Embedding(
            self.tokenizer.vocab_size, 768, padding_idx=self.pad
        ).to(self.device)
        self.embeddingLinear = (
            nn.Linear(768 * self.nBest, 768).to(self.device)
            if self.mode == "align"
            else None
        )

        parameters = list(self.embedding.parameters()) + list(self.model.parameters())
        if self.mode == "align":
            parameters += list(self.embeddingLinear.parameters())
        self.optimizer = AdamW(parameters, lr=lr)

    def forward(
        self,
        input_id,
        attention_mask,
        labels,
    ):
        logging.warning(input_id.shape)
        batch = input_id.shape[0]
        if self.mode == "align":
            input_id = input_id.view(-1, self.nBest)

            # logging.warning(f"input_id.shape:{input_id.shape}")
            aligned_embedding = self.embedding(input_id)  # (L, N, 768)
            # logging.warning(f"aligned_embedding.shape:{aligned_embedding.shape}")
            aligned_embedding = aligned_embedding.view(batch, -1, self.nBest, 768)
            aligned_embedding = aligned_embedding.flatten(start_dim=2)  # (L, 768 * N)
            proj_embedding = self.embeddingLinear(
                aligned_embedding
            )  # (L, 768 * N) -> (L, 768)

            # logging.warning(f"nbest_embedding.shape:{proj_embedding.shape}")
            # logging.warning(f"attention_mask.shape:{attention_mask.shape}")
            # logging.warning(f"labels:{labels.shape}")

            loss = self.model(
                inputs_embeds=proj_embedding,
                attention_mask=attention_mask,
                decoder_input_ids=labels,
                labels=labels,
            ).loss

            logging.warning(f"loss:{loss}")

            return loss

        elif self.mode == "plain":
            assert (
                attention_mask is not None
            ), "Attention Mask will not be produced during plain mode forward"

            loss = self.model(
                input_ids=input_id, attention_mask=attention_mask, labels=labels
            )

            return loss

    def recognize(
        self,
        input_id,
        attention_mask,
    ):
        # input_id : (B, L, N)
        if self.mode == "align":
            aligned_embedding = self.embedding(input_id)  # (L, N, 768)
            aligned_embedding = aligned_embedding.flatten(start_dim=1)  # (L, 768 * N)
            proj_embedding = self.embeddingLinear(
                aligned_embedding
            )  # (L, 768 * N) -> (L, 768)
            nbest_embedding.append(proj_embedding)

            nbest_embedding = pad_sequence(nbest_embedding, batch_first=True).to(
                self.device
            )
            attention_mask = pad_sequence(attention_mask, batch_first=True).to(
                self.device
            )

            output = self.model(
                inputs_embeds=nbest_embedding,
            ).logits

            logging.warning(f"output:{output}")

            return output

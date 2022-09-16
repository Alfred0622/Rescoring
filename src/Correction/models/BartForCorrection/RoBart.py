import torch
import torch.nn as nn
from torch.optim import AdamW
import logging
from transformers import BertTokenizer, BartForConditionalGeneration, BartConfig


class RoBart(nn.Module):
    def __init__(self, device, lr=1e-5):
        nn.Module.__init__(self)
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
        self.model = BartForConditionalGeneration.from_pretrained(
            "fnlp/bart-base-chinese"
        ).to(self.device)

        # self.config = BartConfig.from_pretrained("fnlp/bart-base-chinese")
        # embedding_weight = self.model.model.encoder.embed_tokens.weight.data.clone()
        # decoder_embedding = torch.nn.Embedding(self.config.vocab_size, self.config.d_model, self.config.pad_token_id)
        # decoder_embedding.weight.data = embedding_weight
        # self.model.model.decoder.set_input_embeddings(
        #     decoder_embedding
        # )

        self.optimizer = AdamW(self.model.parameters(), lr = lr)

        logging.warning(self.model)

    def forward(self, input_id, attention_masks, labels, segments = None):

        loss = self.model(
            input_ids = input_id,
            attention_mask = attention_masks, 
            labels=labels,
            return_dict = True
        ).loss

        return loss

    def recognize(self, input_id, attention_masks, segments = None ,max_lens=50):
        
        output = self.model.generate(
            input_ids = input_id,
            attention_mask = attention_masks,
            top_k = 5,
            max_length=max_lens,
        )

        return output

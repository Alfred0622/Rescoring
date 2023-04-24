from numpy import place
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
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
    BartTokenizer,
    BartForConditionalGeneration,
    BartModel,
    AutoConfig
)
from torch.nn.utils.rnn import pad_sequence
from models.nBestAligner.nBestAlign import align, alignNbest

class nBestAlignBart(nn.Module):
    def __init__(
        self,
        device,
        nBest,
        pretrain_name
    ):
        nn.Module.__init__(self)
        self.device = device
        self.nBest = nBest
        self.model = BartForConditionalGeneration.from_pretrained(
            pretrain_name
        ).to(self.device)

        self.linear = nn.Linear(768 * self.nBest, 768).to(self.device)
    
    def forward(self, input_ids, attention_mask):
        """
        input_ids: [B, L ,nBest]
        attention_mask: [B, L]
        """
        aligned_embedding = self.model.share(input_ids) # [B,L ,nBest, 768]
        align_embedding = aligned_embedding.view(input_ids.shape[0], 768 * self.nBest, -1)
        
        aligned_embedding = self.linear(aligned_embedding) # [B, nBest * 768, L] -> [B, 768, L]

        output = self.model(
            inputs_embeds = align_embedding,
            attention_mask = attention_mask,
            return_dict = True
        )

        return output

class nBestTransformer(nn.Module):
    def __init__(
        self,
        nBest,
        device,
        lr=1e-5,
        align_embedding=512,
        dataset = 'aishell',
        from_pretrain = True
    ):
        if (not from_pretrain):
            print("Not From Pretrain")
        else:
            print("From Pretrain")

        nn.Module.__init__(self)
        self.device = device
        self.embedding_dim = align_embedding
        
        if (dataset in ['aishell', 'aishell2', 'aishell_nbest']):
            self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
            if (from_pretrain):
                self.model = BartForConditionalGeneration.from_pretrained(
                    "fnlp/bart-base-chinese"
                ).to(self.device)
            else:
                config = AutoConfig.from_pretrained("fnlp/bart-base-chinese")
                self.model = BartForConditionalGeneration(config = config).to(self.device)
        elif (dataset in ['tedlium2', 'librispeech']):
            self.tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-base')
            if (from_pretrain):
                self.model = BartForConditionalGeneration.from_pretrained(
                    f'facebook/bart-base'
                ).to(self.device)
            else:
                config = AutoConfig.from_pretrained(f'facebook/bart-base')
                self.model = BartForConditionalGeneration(config = config).to(self.device)

        self.nBest = nBest

        self.vocab_size = self.tokenizer.vocab_size
        self.pad = self.tokenizer.convert_tokens_to_ids("[PAD]")

        self.model.config.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(
            "[CLS]"
        )

        self.embedding = nn.Embedding(
            num_embeddings = self.model.config.vocab_size, 
            embedding_dim = align_embedding, 
            padding_idx=self.pad
        ).to(self.device)

        self.embeddingLinear = (
            nn.Linear(align_embedding * self.nBest, 768).to(self.device)
        )

        parameters = list(self.embedding.parameters()) + list(self.model.parameters())
        parameters = parameters + list(self.embeddingLinear.parameters())
        
        self.optimizer = AdamW(parameters, lr=lr)

    def forward(
        self,
        input_id,
        attention_mask,
        labels,
    ):
            
        # print(f'input_ids:{input_id.shape} ')
        aligned_embedding = self.embedding(input_id)  # (L, N, 768)
        # print(f'align_embedding:{aligned_embedding.shape}')
        # logging.warning(f'aligned_embedding.shape:{aligned_embedding.shape}')
        aligned_embedding = aligned_embedding.flatten(start_dim=2)  # (L, 768 * N)
        # logging.warning(f'flattened aligned_embedding.shape:{aligned_embedding.shape}')
        # print(f'aligned_embedding:{aligned_embedding.shape} ')
        proj_embedding = self.embeddingLinear(
            aligned_embedding
        )  # (L, 768 * N) -> (L, 768)

        # labels[labels == 0] = -100

        loss = self.model(
            inputs_embeds=proj_embedding,
            attention_mask=attention_mask,
            labels=labels,
        ).loss

        return loss

    def recognize(
        self,
        input_id,
        attention_mask,
        max_lens,
    ):
        # input_id : (B, L, N)

        batch_size = input_id.shape[0]

        # input_id = input_id.view(-1, self.nBest)
        aligned_embedding = self.embedding(input_id)  # (L, N, 768)
        # aligned_embedding = aligned_embedding.view(
        #     batch, -1, self.nBest, self.embedding_dim
        # )
        aligned_embedding = aligned_embedding.flatten(start_dim=2)  # (L, 768 * N)
        proj_embedding = self.embeddingLinear(
            aligned_embedding
        )  # (L, 768 * N) -> (L, 768)

        decoder_ids = torch.tensor(
            [self.model.config.decoder_start_token_id for _ in range(batch_size)], 
            dtype = torch.int64
        ).unsqueeze(1).to(self.device)
    


        output = self.model.generate(
            inputs_embeds=proj_embedding,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_ids,
            num_beams=5,
            max_length=max_lens,
        )

        return output

    def load_state_dict(self, checkpoint):
        """
        checkpoint:{
            "model",
            "embedding",
            "linear"
        }
        """
        self.model.load_state_dict(checkpoint['model'])
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.embeddingLinear.load_state_dict(checkpoint['linear'])
        
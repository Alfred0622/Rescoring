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
import sys
sys.path.append("../")
from src_utils.getPretrainName import getBartPretrainName

class nBestAlignBart(nn.Module):
    def __init__(
        self,
        args,
        train_args,
        tokenizer,
        **kwargs
    ):
        super().__init__()

        pretrain_name = getBartPretrainName(args['dataset'])
        self.nBest = int(args['nbest'])
        self.model = BartForConditionalGeneration.from_pretrained(
            pretrain_name
        )
        self.embeddding_dim = 768
        # self.pad = tokenizer.convert_tokens_to_ids("[PAD]")
        # self.embedding = nn.Embedding(
        #     num_embeddings = self.model.config.vocab_size, 
        #     embedding_dim = self.embeddding_dim, 
        #     padding_idx=self.pad
        # )

        self.alignLinear = torch.nn.Sequential(
            nn.Linear(self.embeddding_dim * self.nBest, 768),
            # nn.ReLU()
        )
    
    def forward(self, input_ids, attention_mask, labels):
        """
        input_ids: [B, L ,nBest]
        attention_mask: [B, L]
        """
        aligned_embedding = self.model.model.shared(input_ids) # [B,L ,nBest, 768]
        aligned_embedding = aligned_embedding.view(input_ids.shape[0],  self.embeddding_dim * self.nBest, -1).transpose(1,2)
        # aligned_embedding = self.embedding(input_ids)
        # aligned_embedding = aligned_embedding.flatten(start_dim=2)
        aligned_embedding = self.alignLinear(aligned_embedding) # [B, nBest * 768, L] -> [B, 768, L]

        output = self.model(
            inputs_embeds = aligned_embedding,
            attention_mask = attention_mask,
            labels = labels,
            return_dict = True,
        ).loss

        return output

    def recognize(self, input_ids, attention_mask,num_beams = 1 ,max_lens = 150):
        batch_size = input_ids.shape[0]
        aligned_embedding = self.model.model.shared(input_ids)  # (L, N, 768)
        aligned_embedding = aligned_embedding.view(input_ids.shape[0],  self.embeddding_dim * self.nBest, -1).transpose(1,2)  # (L, 768 * N)
        # aligned_embedding = self.embedding(input_ids)
        # aligned_embedding = aligned_embedding.flatten(start_dim=2)
        proj_embedding = self.alignLinear(
            aligned_embedding
        )  # (L, 768 * N) -> (L, 768)

        # print(f'proj_embedding:{proj_embedding.shape}')

        decoder_ids = torch.tensor(
            [self.model.config.decoder_start_token_id for _ in range(batch_size)], 
            dtype = torch.int64
        ).unsqueeze(-1).to(input_ids.device)

        # decoder_ids = torch.tensor(
        #     [[] for _ in range(batch_size)], 
        #     dtype = torch.int64
        # ).to(input_ids.device)

        output = self.model.generate(
            inputs_embeds=proj_embedding,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_ids,
            num_beams=num_beams,
            max_length=max_lens,
            # early_stopping = True
        )

        return output

    def parameters(self):
        return list(self.model.parameters()) + list(self.alignLinear.parameters())

    def state_dict(self):
        return{
            "model": self.model.state_dict(),
            "alignLinear": self.alignLinear.state_dict()
        }

    def load_state_dict(self, checkpoint):
        """
        checkpoint:{
            "model",
            "embedding",
            "linear"
        }
        """
        self.model.load_state_dict(checkpoint['model'])
        self.alignLinear.load_state_dict(checkpoint['alignLinear'])


class nBestTransformer(nn.Module):
    def __init__(
        self,
        nBest,
        device,
        lr=1e-5,
        align_embedding=768,
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

        aligned_embedding = self.embedding(input_id)  # (L, N, 768)
        aligned_embedding = aligned_embedding.flatten(start_dim=2)  # (L, 768 * N)
        proj_embedding = self.embeddingLinear(
            aligned_embedding
        )  # (L, 768 * N) -> (L, 768)

        labels[labels == 0] = -100

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

        aligned_embedding = self.embedding(input_id)  # (L, N, 768)

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
    

        
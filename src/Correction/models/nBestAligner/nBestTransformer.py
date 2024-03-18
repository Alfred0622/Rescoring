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
import time

class nBestAlignBart(nn.Module):
    def __init__(
        self,
        args,
        train_args,
        tokenizer = None,
        **kwargs
    ):
        super().__init__()

        pretrain_name = getBartPretrainName(args['dataset'])
        self.nBest = int(args['nbest'])
        self.model = BartForConditionalGeneration.from_pretrained(
            pretrain_name
        )
        self.embeddding_dim = 768

        self.extra_embedding = train_args['extra_embedding']
        if (train_args['extra_embedding']):
            assert (tokenizer is not None), f"tokenizer need to be given"
            self.pad = tokenizer.pad_token_id
            self.embedding = nn.Embedding(
                num_embeddings = self.model.config.vocab_size, 
                embedding_dim = self.embeddding_dim, 
                padding_idx=self.pad
            )
        self.align_layer = int(train_args['align_layer'])
        print(f"Layer of aligned linear:{self.align_layer}")
        assert(0 < int(self.align_layer) and int(self.align_layer) <= 2), f"Number of align layer should be 1 or 2"

        if (self.align_layer == 1):
            print(f'{self.embeddding_dim} * {self.nBest} = {self.embeddding_dim * self.nBest}')
            self.alignLinear = torch.nn.Sequential(
                nn.Dropout(p = 0.1),
                nn.Linear(self.embeddding_dim * self.nBest, 768),
            )
        elif (self.align_layer == 2):
            print(f'two layer')
            self.alignLinear = torch.nn.Sequential(
                nn.Dropout(p = 0.3),
                nn.Linear(self.embeddding_dim * self.nBest, 1024),
                nn.ReLU(),
                nn.Dropout(p = 0.3),
                nn.Linear(1024, 768),
            )
    
    def forward(self, input_ids, attention_mask, labels):
        """
        input_ids: [B, L ,nBest]
        attention_mask: [B, L]
        """
        if (self.extra_embedding):
            aligned_embedding = self.embedding(input_ids)
            aligned_embedding = torch.flatten(aligned_embedding, start_dim=2)

        else:
            aligned_embedding = self.model.model.shared(input_ids)  # (L, N, 768)
            aligned_embedding = torch.flatten(aligned_embedding, start_dim = 2)# (L, 768 * N)
        
        aligned_embedding = self.alignLinear(aligned_embedding)

        output = self.model(
            inputs_embeds = aligned_embedding,
            attention_mask = attention_mask,
            labels = labels,
            return_dict = True,
        ).loss

        return output

    def recognize(self, input_ids, attention_mask,num_beams = 5 ,max_lens = 150):
        batch_size = input_ids.shape[0]
        decoder_ids = torch.tensor(
            [self.model.config.decoder_start_token_id for _ in range(batch_size)], 
            dtype = torch.int64
        ).unsqueeze(-1).to(input_ids.device)
        elapsed_time = 0.0
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        if (self.extra_embedding):
            
            # t0 = time.time()
            start.record()
            aligned_embedding = self.embedding(input_ids)
            aligned_embedding = aligned_embedding.flatten(start_dim=2)
            end.record()
            torch.cuda.synchronize()
            # torch.cuda.synchronize()
            # t1 = time.time()

        
        else:
            # torch.cuda.synchronize()
            # t0 = time.time()
            start.record()
            aligned_embedding = self.model.model.shared(input_ids)  # (L, N, 768)
            aligned_embedding = torch.flatten(aligned_embedding, start_dim = 2)# (L, 768 * N)
            end.record()
            torch.cuda.synchronize()
            # t1 = time.time()

        elapsed_time += start.elapsed_time(end)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # torch.cuda.synchronize()
        # t0 = time.time()
        start.record()
        proj_embedding = self.alignLinear(
            aligned_embedding
        )  # (L, 768 * N) -> (L, 768)
        end.record()
        torch.cuda.synchronize()
        elapsed_time += start.elapsed_time(end)

        # print(f'proj_embedding:{proj_embedding.shape}')



        # decoder_ids = torch.tensor( 
        #     [[] for _ in range(batch_size)], 
        #     dtype = torch.int64
        # ).to(input_ids.device)
        # torch.cuda.synchronize()
        # t0 = time.time()
        elapsed_time += start.elapsed_time(end)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = self.model.generate(
            inputs_embeds=proj_embedding,
            attention_mask=attention_mask,
            # decoder_input_ids=decoder_ids,
            num_beams=num_beams,
            max_length=max_lens,
            # early_stopping = True
        )
        end.record()
        torch.cuda.synchronize()
        elapsed_time += start.elapsed_time(end)

        return output, elapsed_time

    def parameters(self):
        return list(self.model.parameters()) + list(self.alignLinear.parameters())

    def state_dict(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "alignLinear": self.alignLinear.state_dict()
        }
        if (self.extra_embedding):
            checkpoint['embedding'] = self.embedding
        return checkpoint

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
        if (self.extra_embedding):
            self.embedding.load_state_dict(checkpoint['embedding'].state_dict())
    
    def show_param(self):
        print(sum(p.numel() for p in self.parameters()))


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
    

        
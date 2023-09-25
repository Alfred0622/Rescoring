import torch
import numpy as np


class MaxPooling(torch.nn.Module):
    def __init__(self, noCLS=True, noSEP=False):
        super().__init__()
        self.noCLS = noCLS
        self.noSEP = noSEP

    def forward(self, input_ids, input_embeds, attention_mask):
        """
        # input_embeds: (B, L, D)
        # attention_mask: (B, L)
        """

        if self.noCLS:
            attention_mask[input_ids == 101] = 0

        if self.noSEP:
            attention_mask[input_ids == 102] = 0

        embeds = input_embeds.clone()

        # print(f"embeds:{embeds}")
        embeds[attention_mask == 0] = -1e9
        # print(f"embeds after masking:{embeds}")

        max_embedding, _ = torch.max(embeds.clone(), dim=1)
        # print(f"max_embedding:{max_embedding.shape}")

        return max_embedding


class AvgPooling(torch.nn.Module):
    def __init__(self, noCLS=True, noSEP=False):
        super().__init__()
        self.noCLS = noCLS
        self.noSEP = noSEP

    def forward(self, input_ids, input_embeds, attention_mask):
        """
        # input_ids: (B, L, D)
        # attention_mask: (B, L)
        """
        if self.noCLS:
            attention_mask[input_ids == 101] = 0

        if self.noSEP:
            attention_mask[input_ids == 102] = 0

        seq_length = torch.sum(attention_mask, dim=-1).unsqueeze(-1)
        seq_length[seq_length <= 0] = 1
        mean_embedding = (
            torch.sum(
                input_embeds.clone() * attention_mask.clone().unsqueeze(-1),
                dim=1,
            ).clone()
            / seq_length
        )
        return mean_embedding


class MinPooling(torch.nn.Module):
    def __init__(self, noCLS=True, noSEP=False):
        super().__init__()
        self.noCLS = noCLS
        self.noSEP = noSEP

    def forward(self, input_ids, input_embeds, attention_mask):
        """
        # input_ids: (B, L, D)
        # attention_mask: (B, L)
        """
        if self.noCLS:
            attention_mask[input_ids == 101] = 0

        if self.noSEP:
            attention_mask[input_ids == 102] = 0

        weight = attention_mask.clone()
        weight[weight == 0] = np.Inf
        min_embedding, _ = torch.min(input_embeds * weight, dim=1)

        return min_embedding

import torch

class MaxPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_ids, attention_mask):
        """
        # input_ids: (B, L, D)
        # attention_mask: (B, L)
        """
        max_embedding, _ = torch.max(input_ids * attention_mask.unsqueeze(-1), dim = 1)

        return max_embedding

class AvgPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_ids, attention_mask):
        """
        # input_ids: (B, L, D)
        # attention_mask: (B, L)
        """
        # print(f"input_ids.shape:{input_ids.shape}")
        # print(f"attention_mask.shape:{attention_mask.shape}")

        # print(f"{torch.sum(input_ids * attention_mask.unsqueeze(-1), dim = 1).shape}")
        # print(f"{torch.sum(attention_mask, dim = -1).unsqueeze(-1).shape}")
        mean_embedding = torch.sum(input_ids * attention_mask.unsqueeze(-1), dim = 1) / torch.sum(attention_mask, dim = -1).unsqueeze(-1)
        return mean_embedding

class MinPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_ids, attention_mask):
        """
        # input_ids: (B, L, D)
        # attention_mask: (B, L)
        """
        weight = attention_mask.clone()
        weight[weight == 0] = 1e9
        min_embedding = torch.min(input_ids * weight.unsqueeze(-1), dim = 1)

        return min_embedding
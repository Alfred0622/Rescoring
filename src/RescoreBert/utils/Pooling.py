import torch

class MaxPooling(torch.nn.Module):
    def __init__(self, noCLS = True,  noSEP = False ):
        super().__init__()
        self.noCLS = noCLS
        self.noSEP = noSEP
    
    def forward(self, input_ids, attention_mask):
        """
        # input_ids: (B, L, D)
        # attention_mask: (B, L)
        """

        if (self.noCLS):
            attention_mask[:, 0] = 0
        
        if (self.noSEP):
            attention_mask = torch.roll(attention_mask, -1, -1)
            attention_mask[:, -1] = 0

        max_embedding, _ = torch.max(input_ids * attention_mask.unsqueeze(-1), dim = 1)

        return max_embedding

class AvgPooling(torch.nn.Module):
    def __init__(self, noCLS = True,  noSEP = False ):
        super().__init__()
        self.noCLS = noCLS
        self.noSEP = noSEP
    def forward(self, input_ids, attention_mask):
        """
        # input_ids: (B, L, D)
        # attention_mask: (B, L)
        """
        if (self.noCLS):
            attention_mask[:, 0] = 0
        
        if (self.noSEP):
            attention_mask = torch.roll(attention_mask, -1, -1)
            attention_mask[:, -1] = 0

        mean_embedding = torch.sum(input_ids * attention_mask.unsqueeze(-1), dim = 1) / torch.sum(attention_mask, dim = -1).unsqueeze(-1)
        return mean_embedding

class MinPooling(torch.nn.Module):
    def __init__(self, noCLS = True,  noSEP = False ):
        super().__init__()
        self.noCLS = noCLS
        self.noSEP = noSEP
    def forward(self, input_ids, attention_mask):
        """
        # input_ids: (B, L, D)
        # attention_mask: (B, L)
        """
        if (self.noCLS):
            attention_mask[:, 0] = 0
        
        if (self.noSEP):
            attention_mask = torch.roll(attention_mask, -1, -1)
            attention_mask[:, -1] = 0
        
        weight = attention_mask.clone()
        weight[weight == 0] = 1e9
        min_embedding = torch.min(input_ids * weight.unsqueeze(-1), dim = 1)

        return min_embedding
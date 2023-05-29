import torch

class SoftmaxOverNBest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._softmax = torch.nn.Softmax(dim = -1)
        self._logSoftmax = torch.nn.LogSoftmax(dim = -1)
    
    def forward(self, scores, nBestIndex, log_score = False):
        start_index = 0
        for index in nBestIndex:
            if (not log_score):
                scores[start_index : start_index + index] = self._softmax(scores[start_index : start_index + index].clone())
            else:
                scores[start_index : start_index + index] = self._logSoftmax(scores[start_index : start_index + index].clone())
            
            start_index += index
        
        return scores
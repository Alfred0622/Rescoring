import torch


class SoftmaxOverNBest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._softmax = torch.nn.Softmax(dim=-1)
        self._logSoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, scores, nBestIndex, log_score=False, paddingNbest=False, topk=50):
        start_index = 0
        if len(scores.shape) == 1:
            for index in nBestIndex:
                # print(f"scores:{scores[start_index : start_index + index]}")
                if not log_score:
                    scores[start_index : start_index + index] = self._softmax(
                        scores[start_index : start_index + index].clone()
                    )
                else:
                    scores[start_index : start_index + index] = self._logSoftmax(
                        scores[start_index : start_index + index].clone()
                    )
                if paddingNbest:
                    start_index += topk
                else:
                    start_index += index
        elif len(scores.shape) == 2:
            for index in nBestIndex:
                if not log_score:
                    scores[:, start_index : start_index + index,] = self._softmax(
                        scores[
                            :,
                            start_index : start_index + index,
                        ].clone(),
                    )
                else:
                    scores[:, start_index : start_index + index,] = self._logSoftmax(
                        scores[
                            :,
                            start_index : start_index + index,
                        ].clone(),
                    )
                if paddingNbest:
                    start_index += topk
                else:
                    start_index += index

        return scores

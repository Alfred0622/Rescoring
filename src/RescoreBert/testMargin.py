import torch
from model.ContrastBERT import selfMarginLoss

x = torch.tensor([5])
y = torch.tensor([1, 2, 3, 4, 5, 6])
rank = torch.tensor([[4, 2, 1, 3, 0, 5]])
NBest = [6]
margin_1 = torch.nn.MarginRankingLoss(margin=0.01)
margin_2 = selfMarginLoss(margin=0.01)

x_s = x.expand_as(y)

loss_1 = 0

loss_2 = margin_2(y, NBest, rank)

print(f"Torch ===================")
for r, i in enumerate(rank[0][:-1]):
    pos = y[rank[0][r]]
    neg = y[rank[0][r + 1 :]]
    pos_s = pos.expand_as(neg)

    print(f"pos:{pos}")
    print(f"neg:{neg}")

    loss_1 += margin_1(pos_s, neg, torch.ones(neg.shape))

print(f"Torch:{loss_1}, My:{loss_2}")

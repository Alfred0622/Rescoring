


# def createBatch(sample):
#     token_id = [s[0] + s[1] for s in sample]
#     seg_id = [[0 * len(s[0])] + [1 * len(s[1])] for s in sample]

#     for i, token in enumerate(token_id):
#         token_id[i] = torch.tensor(token)
#         seg_id[i] = torch.tensor(seg_id[i])

#     token_id = pad_sequence(token_id, batch_first=True)
#     seg_id = pad_sequence(seg_id, batch_first=True, padding_value=1)

#     attention_mask = torch.zeros(token_id.shape)
#     attention_mask = attention_mask.masked_fill(token_id != 0, 1)

#     labels = [s[2] for s in sample]

#     for i, label in labels:
#         labels[i] = torch.tensor(label)
#     labels = pad_sequence(labels, batch_first=True, padding_value=-100)

#     return token_id, seg_id, attention_mask, labels
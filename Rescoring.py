import os
from tqdm import tqdm
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from BertForRescring import RescoreBert

epochs = 50
train_batch = 3
test_batch = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class nBestDataset(Dataset):
    def __init__(self, nbest_dict):
        """
        nbest_dict: {token seq : CER}
        """
        self.tokens = torch.tensor(nbest_dict["token"])
        self.seg = torch.tensor(nbest_dict["segment"])
        self.score = torch.tensor(nbest_dict["score"])
        self.cer = torch.tensor(nbest_dict["cer"])
    
    def __getitem__(self, idx):
        return self.tokens[idx],self.seg[idx] , self.score[idx], self.cer[idx]
    def __len__(self):
        return len(self.tokens)

def createBatch(sample):
    tokens = [s[0] for s in sample]
    segs = [s[1] for s in sample]
    scores = [s[2] for s in sample]
    cers = [s[3] for s in sample]

    tokens = pad_sequence(
        tokens,
        batch_first = True
    )

    segs = pad_sequence(
        segs,
        batch_first = True
    )

    masks = torch.zeros(
        tokens.shape,
        dtype = torch.long 
        )
    
    masks = masks.masked_fill(tokens != 0 , 1)
    
    return tokens, segs, masks, scores, cers
    
# load nbest data
train_json = []
dev_json = []
test_json = []

load_name = ['train', 'dev', 'test']
jsons = [train_json, dev_json, test_json]

for data, name in jsons, load_name:
    # os.listdir()
    for i in range(1, 33):
        file_name = f'./data/aishell_{name}/data.{i}.json'
        with open(file_name) as f:
            json_file = json.load(f)

        for utt in json_file['utts']:
            n_best_list = []
            for wav_id in utt:
                hyp2score=dict()
                for output in json_file['utts'][wav_id]['output']:
                    # get the token, segment id, attention mask
                    hyp2score[output['rec_tokenid']] = output['score']
                n_best_list.append(hyp2score)
            data.append(n_best_list)

train_set = nBestDataset(train_json)
dev_set = nBestDataset(dev_json)
test_set = nBestDataset(test_json)

train_loader = DataLoader(
    dataset = train_set,
    batch_size = train_batch,
    shuffle = True,
    collate_fn=createBatch
)
dev_loader =DataLoader(
    dataset = train_set,
    batch_size = test_batch,
    shuffle = True,
    collate_fn=createBatch
)
test_loader = DataLoader(
    dataset = train_set,
    batch_size = test_batch,
    shuffle = True,
    collate_fn=createBatch
)

accumgrad = 8

# declear model
model = RescoreBert()
optimizer = None


# training
logging.warning(f'training')
optimizer.zero_grad()
model.train()
for e in range(epochs):
    accum_loss = 0.0
    for n, token, seg, score, cer in enumerate(tqdm(train_loader)):
        token = token.to(device)
        seg = seg.to(device)
        score = score.to(device)
        cer = cer.to(device)
        
        
        accum_loss += model(token, seg, score)
        if ((n + 1) % accum_loss == 0):
            accum_loss.backward()
            optimizer.step()
            accum_loss = 0.0
            optimizer.zero_grad()
    
    logging.warning(f'epochs: {e + 1}: validation')
    for n, token, seg, score, cer in enumerate(tqdm(dev_loader)):
        token = token.to(device)
        seg = seg.to(device)
        score = score.to(device)
        cer = cer.to(device)
    
    
    
    

# recognizing
model.eval()
logging.warning(f'recognizing')
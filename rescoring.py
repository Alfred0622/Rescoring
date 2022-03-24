import os
from tqdm import tqdm
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from BertForRescring.RescoreBert import RescoreBert

epochs = 1
train_batch = 3
test_batch = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
load_state_dict = False

FORMAT = '%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s'
logging.basicConfig(level=logging.INFO, filename='./log/train.log', filemode='w', format=FORMAT)

class nBestDataset(Dataset):
    def __init__(self, nbest_list):
        """
        nbest_dict: {token seq : CER}
        """
        self.data = nbest_list
    
    def __getitem__(self, idx):
        return self.data[idx]['token'], \
               self.data[idx]['segment'], \
               self.data[idx]['score'], \
               self.data[idx]['err']
    def __len__(self):
        return len(self.data)

def createBatch(sample):
    tokens = []
    for s in sample:
        tokens += s[0]
        
    segs = []
    for s in sample:
        segs += s[1]

    scores = []
    for s in sample:
        scores += s[2]
    
    cers = []
    for s in sample:
        cers += s[3]
    
    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    for i, s in enumerate(segs):
        segs[i] = torch.tensor(s)
    

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


    return tokens, segs, masks, torch.tensor(scores), torch.tensor(cers)
    
# load nbest data
train_json = None
dev_json = None
test_json = None

load_name =  ['dev', 'test'] 
jsons = [dev_json, test_json]

print(f'Prepare data')
for name in  load_name:
    file_name = f'./data/aishell_{name}/token.json'
    with open(file_name) as f:
        if (name == 'dev'):
            dev_json = json.load(f)
        elif (name == 'test'):
            test_json = json.load(f)

# train_set = nBestDataset(train_json)
dev_set = nBestDataset(dev_json)
test_set = nBestDataset(test_json)


# train_loader = DataLoader(
#     dataset = train_set,
#     batch_size = train_batch,
#     shuffle = True,
#     collate_fn=createBatch
# )
dev_loader =DataLoader(
    dataset = dev_set,
    batch_size = test_batch,
    shuffle = True,
    collate_fn=createBatch
)
test_loader = DataLoader(
    dataset = test_set,
    batch_size = test_batch,
    shuffle = True,
    collate_fn=createBatch
)

accumgrad = 8

# declear model
model = RescoreBert(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
train_data = dev_set

print("domain adaption")
optimizer.zero_grad()
for e in range(3):
    model.train()
    accum_loss = 0.0
    for n, data in enumerate(tqdm(train_data)):
        token, seg, mask, score, cer = data
        token = token.to(device)
        seg = seg.to(device)
        mask = mask.to(device)
        score = score.to(device)
        cer = cer.to(device)

        logging.warning(f'token:{token.shape}')
        
        accum_loss += model.adaption(token, seg, mask, score , cer)

        if ((n + 1) % accum_loss == 0):
            accum_loss.backward()
            optimizer.step()
            accum_loss = 0.0
            optimizer.zero_grad()

# training
print(f'training...')
optimizer.zero_grad()


for e in range(epochs):
    model.train()
    accum_loss = 0.0
    for n, data in enumerate(tqdm(train_data)):
        token, seg, mask, score, cer = data
        token = token.to(device)
        seg = seg.to(device)
        mask = mask.to(device)
        score = score.to(device)
        cer = cer.to(device)

        logging.warning(f'token:{token.shape}')
        
        accum_loss += model(token, seg, mask, score , cer)

        if ((n + 1) % accum_loss == 0):
            accum_loss.backward()
            optimizer.step()
            accum_loss = 0.0
            optimizer.zero_grad()
            
    model.save_pretrained(f'checkpoint/checkpoint_{n+1}')
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for n, token, seg, mask, score, cer in enumerate(tqdm(dev_loader)):
            token = token.to(device)
            seg = seg.to(device)
            mask = mask.to(device)
            score = score.to(device)
            cer = cer.to(device)

            val_loss += model(token, seg , mask, score, cer)
    logging.warning(f'epoch :{n + 1}, validation_loss:{val_loss}')
    

# recognizing
logging.warning(f'recognizing')
if (load_state_dict):
    last_checkpoint = len(os.listdir('./checkpoint/'))
    model = RescoreBert.from_pretrained(f'checkpoint/checkpoint_{last_checkpoint}')

model.eval()

recog_set = ['dev', 'test']
with torch.no_grad():
    for data in recog_set:
        recog_dict = dict()
        recog_dict['utts'] = dict()
        for n, token, seg, mask, score, cer in enumerate(tqdm(dev_loader)):
            token = token.to(device)
            seg = seg.to(device)
            mask = mask.to(device)
            score = score.to(device)
            cer = cer.to(device)

            best_hyp = model.recognize(token, seg, mask, score)
            token_list = [str(t) for t in best_hyp]
            recog_dict['utts'][f'{data}_{n}'] = dict()
            recog_dict['utts'][f'{data}_{n}']['output'].append({
                    'rec_text': "".join(token_list),
                    'rec_token': " ".join(token_list),
                    # 'text': groundtruth 
                }
        )

        with open(f'data/aishell_{data}/rescore_data.json') as f:
            json.dump(recog_dict, f, ensure_ascii=False)
    






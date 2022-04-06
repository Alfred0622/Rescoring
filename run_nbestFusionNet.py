import os
from tqdm import tqdm
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nBestFusionNet.fusionNet import fusionNet

"""Basic setting"""
epochs = 30
train_batch = 64
test_batch = 1
# device = 'cpu' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
accumgrad = 1
print_loss = 200

stage = 1

""""""
FORMAT = '%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s'
logging.basicConfig(level=logging.INFO, filename=f'./log/nBestFusionNet/train.log', filemode='w', format=FORMAT)


""" Methods """

class nBestDataset(Dataset):
    def __init__(self, nbest_list):
        """
        nbest_dict: {token seq : CER}
        """
        self.data = nbest_list
    
    def __getitem__(self, idx):
        return self.data[idx]['token'],\
               self.data[idx]['segment'],\
               self.data[idx]['ref_token'],\
               self.data[idx]['ref_seg'],\
               self.data[idx]['ref'],\
               
    def __len__(self):
        return len(self.data)

# for training dataloader
def createBatch(sample):
    tokens = []
    label = []
    segs = []
    for s in sample:
        label_index = torch.randint(low = 0, high = len(s[0]), size = (1,) ).item()
        s[0][label_index] = s[2]
        s[1][label_index] = s[3]
        tokens += s[0]
        segs += s[1]
        label.append(label_index)
    
    
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

    label = torch.tensor(label)
 
    return tokens, segs, masks, label

# for recognition dataloader
def recogBatch(sample):
    tokens = []
    segs = []

    for s in sample:
        tokens.append(s[0])
        segs.append(s[1])
    
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

    ref = [s[5] for s in sample]

    return tokens, segs, masks, ref

print(f'Prepare data')
train_json = None
dev_json = None
test_json = None

load_name =  ['train', 'dev', 'test'] 

for name in  load_name:
    file_name = f'./data/aishell_{name}/token.json'
    with open(file_name) as f:
        if (name == 'train'):
            train_json = json.load(f)
        elif (name == 'dev'):
            dev_json = json.load(f)
        elif (name == 'test'):
            test_json = json.load(f)

nBest = len(train_json[0]['token'])

train_set = nBestDataset(train_json)
dev_set = nBestDataset(dev_json)
test_set = nBestDataset(test_json)


"""Training Dataloader"""

train_loader = DataLoader(
    dataset = train_set,
    batch_size = train_batch,
    collate_fn= createBatch
)

valid_loader = DataLoader(
    dataset = dev_set,
    batch_size = train_batch,
    collate_fn= createBatch
) 

dev_loader =DataLoader(
    dataset = dev_set,
    batch_size = test_batch,
    collate_fn= recogBatch
)
test_loader = DataLoader(
    dataset = test_set,
    batch_size = test_batch,
    collate_fn= recogBatch
)



"""Init model""" 
logging.warning(f'device:{device}')
device = torch.device(device)
model = fusionNet(device = device, num_nBest=nBest)
train_optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

scoring_set = ['train', 'dev']

if (stage <= 1):
    """training"""
    
    train_optimizer.zero_grad()
    
    last_val = 1e8
    for e in range(epochs):
        model.train()
        accum_loss = 0.0
        logging_loss = 0.0
        for n, data in enumerate(tqdm(train_loader)):
            # if (n < 16000):
            #     continue
            token, seg, mask, label = data
            token = token.to(device)
            seg = seg.to(device)
            mask = mask.to(device)
            label = label.to(device)
            
            loss = model(token, seg, mask, label)
            loss = loss / accumgrad
            loss.backward()

            logging_loss += loss.clone().detach().cpu()

            if ((n + 1) % accumgrad == 0 or (n + 1) == len(train_loader)):
                train_optimizer.step()
                train_optimizer.zero_grad()

            if ((n + 1) % print_loss == 0):
                logging.warning(f'Training epoch:{e + 1} step:{n + 1}, training loss:{logging_loss}')
                logging_loss = 0.0

        torch.save(model.state_dict(), f"./checkpoint/nBestFusionNet/checkpoint_train_{e + 1}.pt")
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for n, data in enumerate(tqdm(valid_loader)):
                token, seg, mask, score, cer, pll = data
                token = token.to(device)
                seg = seg.to(device)
                mask = mask.to(device)
                score = score.to(device)
                cer = cer.to(device)
                pll = pll.to(device)

                val_loss += model(token, seg , mask, score, cer, pll)
            val_loss = val_loss / len(dev_loader)

        logging.warning(f'epoch :{e + 1}, validation_loss:{val_loss}')
        if (last_val - val_loss < 1e-4):
            print('early stop')
            logging.warning(f'early stop')
            break
        last_val = val_loss
        

# recognizing
if (stage <= 2):
    print(f'recognizing')
    if (stage == 2):
        print(f'using checkpoint: ./checkpoint/nBestFusionNet/checkpoint_train_{epochs}.pt')
        model_args = torch.load(f'./checkpoint/nBestFusionNet/checkpoint_train_{epochs}.pt')
        model.load_state_dict(model_args)

    model.eval()
    recog_set = ['dev', 'test']
    recog_data = None
    with torch.no_grad():
        for task in recog_set:
            print(f'recogizing: {task}')
            if (task == 'dev'):
                recog_data = dev_loader
            elif (task == 'test'):
                recog_data = test_loader

            recog_dict = dict()
            recog_dict['utts'] = dict()
            for n, data in enumerate(tqdm(recog_data)):
                token, seg, mask, ref = data
                token = token.to(device)
                seg = seg.to(device)
                mask = mask.to(device)

                best_hyp = model.recognize(token, seg, mask)
                token_list = [str(t) for t in best_hyp]  # remove [CLS] and [SEP]
                ref_list = [str(t) for t in ref[0][5:-5]]
                recog_dict['utts'][f'{task}_{n}'] = dict()
                recog_dict['utts'][f'{task}_{n}']['output'] = {
                        'rec_text': "".join(token_list),
                        'rec_token': " ".join(token_list),
                        "text": "".join(ref_list),
                        "text_token": " ".join(ref_list)
                    }
            
            with open(f'data/aishell_{task}/nBestFusionNet/rescore_data.json', 'w') as f:
                json.dump(recog_dict, f, ensure_ascii=False, indent = 2)

    print('Finish')
    






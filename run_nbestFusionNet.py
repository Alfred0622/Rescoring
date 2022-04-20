import os
from tqdm import tqdm
import json
import yaml
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nBestFusionNet.fusionNet import fusionNet
import random

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

FORMAT = '%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s'
logging.basicConfig(level=logging.INFO, filename=f'./log/nBestFusionNet/train.log', filemode='w', format=FORMAT)
"""Basic setting"""
config = './config/nBestFusionNet.yaml'
stage = 0
train_args = dict()
recog_args = dict()

with open(config, 'r') as f:
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    stage = conf['stage']
    train_args = conf['train']
    recog_args = conf['recog']

epochs = train_args['epoch']
train_batch = train_args['train_batch']
test_batch = recog_args['batch']

accumgrad = train_args['accumgrad']
print_loss = train_args['print_loss']

# device = 'cpu' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True
""""""

""" Methods """

class nBestDataset(Dataset):
    def __init__(self, nbest_list):
        """
        nbest_dict: {token seq : CER}
        """
        self.data = nbest_list
    
    def __getitem__(self, idx):
        return self.data[idx]['token'],\
               self.data[idx]['ref_token'],\
               self.data[idx]['ref'],\
               
    def __len__(self):
        return len(self.data)

# for training dataloader
def createBatch(sample):
    tokens = []
    label = []
    for s in sample:
        if (s[1] in s[0]):
            label_index = s[0].index(s[1])
            label.append(label_index)
        else:
            label_index = torch.randint(low = 0, high = len(s[0]), size = (1,) ).item()
            s[0][label_index] = s[1]
            label.append(label_index)
        
        assert (s[1] in s[0]), "no reference in training batch"
        tokens += s[0]

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    
    tokens = pad_sequence(
        tokens,
        batch_first = True
    )

    masks = torch.zeros(
        tokens.shape,
        dtype = torch.long 
    )
    masks = masks.masked_fill(tokens != 0 , 1)

    label = torch.tensor(label)
 
    return tokens, masks, label

# for recognition dataloader
def recogBatch(sample):
    tokens = []
    segs = []

    for s in sample:
        tokens += s[0]
    
    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)

    
    tokens = pad_sequence(
        tokens,
        batch_first = True
    )

    masks = torch.zeros(
        tokens.shape,
        dtype = torch.long 
    )
    masks = masks.masked_fill(tokens != 0 , 1)

    ref = [s[2] for s in sample]

    return tokens, masks, ref

print(f'Prepare data')
train_json = None
dev_json = None
test_json = None

load_name =  ['train', 'dev', 'test'] 

for name in  load_name:
    file_name = f'./data/aishell_{name}/token_10best.json'
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
    collate_fn= createBatch,
    shuffle=True,
    pin_memory=True,
    num_workers=4
)

valid_loader = DataLoader(
    dataset = dev_set,
    batch_size = train_batch,
    collate_fn= createBatch
) 

recog_train_loader=DataLoader(
    dataset = train_set,
    batch_size = test_batch,
    collate_fn= recogBatch
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


scoring_set = ['train', 'dev']

if (stage <= 1):
    """training"""
    
    model.optimizer.zero_grad()
    
    last_val = 1e8
    for e in range(epochs):
        model.train()
        accum_loss = 0.0
        logging_loss = 0.0
        for n, data in enumerate(tqdm(train_loader)):
            # if (n < 16000):
            #     continue
            token, mask, label = data
            token = token.to(device)
            mask = mask.to(device)
            label = label.to(device)
            
            loss = model(token, mask, label)
            loss = loss / accumgrad
            loss.backward()

            logging_loss += loss.clone().detach().cpu()

            if ((n + 1) % accumgrad == 0 or (n + 1) == len(train_loader)):
                model.optimizer.step()
                model.scheduler.step()

            if ((n + 1) % print_loss == 0):
                logging.warning(f'Training epoch:{e + 1} step:{n + 1}, training loss:{logging_loss}')
                logging_loss = 0.0

        torch.save(model.state_dict(), f"./checkpoint/nBestFusionNet/checkpoint_train_{e + 1}.pt")
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for n, data in enumerate(tqdm(valid_loader)):
                token, mask, label = data
                token = token.to(device)
                mask = mask.to(device)
                label = label.to(device)

                val_loss += model(token, mask, label)
            val_loss = val_loss / len(dev_loader)
            logging.warning(f'epoch :{e + 1}, validation_loss:{val_loss}')
        # if (last_val - val_loss < 1e-4):
        #     print('early stop')
        #     logging.warning(f'early stop')
        #     break
        # last_val = val_loss
        

# recognizing
if (stage <= 2):
    print(f'recognizing')
    if (stage == 2):

        print(f'using checkpoint: ./checkpoint/nBestFusionNet/checkpoint_train_2.pt')
        model_args = torch.load(f'./checkpoint/nBestFusionNet/checkpoint_train_2.pt')
        model.load_state_dict(model_args)

    model.eval()
    recog_set = ['train', 'dev', 'test']
    recog_data = None
    with torch.no_grad():
        for task in recog_set:
            print(f'recogizing: {task}')
            if (task == 'train'):
                recog_data = recog_train_loader
            if (task == 'dev'):
                recog_data = dev_loader
            elif (task == 'test'):
                recog_data = test_loader

            recog_dict = dict()
            recog_dict['utts'] = dict()
            for n, data in enumerate(tqdm(recog_data)):
                token, mask, ref = data
                token = token.to(device)
                mask = mask.to(device)

                best_hyp = model.recognize(token, mask)
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
    






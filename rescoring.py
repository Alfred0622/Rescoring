import os
from tqdm import tqdm
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from BertForRescring.RescoreBert import RescoreBert

"""Basic setting"""
adapt_epoch = 5
epochs = 10
adaption_batch = 256
train_batch = 64
test_batch = 1
# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
accumgrad = 1
print_loss = 200

use_MWER = False
use_MWED = False

stage = 4

adaption_mode = 'sequence'
if (adaption_mode is 'sequence'):
    adaption_batch = 32
""""""

training = "MD"
if (use_MWER):
    print('using MWER')
    training = "MWER"
elif (use_MWED):
    print('using MWED')
    training = "MWED"


FORMAT = '%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s'
logging.basicConfig(level=logging.INFO,
                    filename=f'./log/{training}_train.log', filemode='w', format=FORMAT)

model_args = {'training': None, 'state_dict': None,
              'adapt_optimizer': None, 'train_optimizer': None, 'last_val_loss': None}
model_args['training'] = training

""" Methods """


class adaptionData(Dataset):
    def __init__(self, nbest_list):
        self.data = nbest_list

    def __getitem__(self, idx):
        return self.data[idx]['ref_token'], self.data[idx]['ref_seg']

    def __len__(self):
        return len(self.data)


class nBestDataset(Dataset):
    def __init__(self, nbest_list):
        """
        nbest_dict: {token seq : CER}
        """
        self.data = nbest_list

    def __getitem__(self, idx):
        return self.data[idx]['token'],\
            self.data[idx]['segment'],\
            self.data[idx]['score'],\
            self.data[idx]['err'],\
            self.data[idx]['pll']

        #    self.data[idx]['name'],\

    def __len__(self):
        return len(self.data)


class rescoreDataset(Dataset):
    def __init__(self, nbest_list):
        """
        nbest_dict: {token seq : CER}
        """
        self.data = nbest_list

    def __getitem__(self, idx):
        return self.data[idx]['name'],\
            self.data[idx]['token'],\
            self.data[idx]['segment'],\
            self.data[idx]['score'],\
            self.data[idx]['ref']

    def __len__(self):
        return len(self.data)

# Fro domain adaption


def adaptionBatch(sample):
    tokens = [torch.tensor(s[0]) for s in sample]
    segs = [torch.tensor(s[1]) for s in sample]

    tokens = pad_sequence(
        tokens,
        batch_first=True
    )

    segs = pad_sequence(
        segs,
        batch_first=True
    )

    masks = torch.zeros(
        tokens.shape,
        dtype=torch.long
    )
    masks = masks.masked_fill(tokens != 0, 1)

    return tokens, segs, masks

# pll scoring & recognizing


def scoringBatch(sample):
    name = [s[0] for s in sample]

    tokens = []
    for s in sample:
        tokens += s[1]

    segs = []
    for s in sample:
        segs += s[2]

    scores = []
    for s in sample:
        scores += s[3]

    ref = [s[4] for s in sample]

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    for i, s in enumerate(segs):
        segs[i] = torch.tensor(s)

    tokens = pad_sequence(
        tokens,
        batch_first=True
    )

    segs = pad_sequence(
        segs,
        batch_first=True
    )

    masks = torch.zeros(
        tokens.shape,
        dtype=torch.long
    )
    masks = masks.masked_fill(tokens != 0, 1)

    return name[0], tokens, segs, masks, torch.tensor(scores), ref

#  MD distillation


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

    pll = [s[4] for s in sample]
    for p in pll:
        assert (len(p) == 10), f'illegal pll:{p}'
    pll = torch.tensor(pll)

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    for i, s in enumerate(segs):
        segs[i] = torch.tensor(s)

    tokens = pad_sequence(
        tokens,
        batch_first=True
    )

    segs = pad_sequence(
        segs,
        batch_first=True
    )

    masks = torch.zeros(
        tokens.shape,
        dtype=torch.long
    )
    masks = masks.masked_fill(tokens != 0, 1)

    return tokens, segs, masks, torch.tensor(scores), torch.tensor(cers), pll


print(f'Prepare data')
train_json = None
dev_json = None
test_json = None

load_name = ['train', 'dev', 'test']

for name in load_name:
    file_name = f'./data/aishell_{name}/token.json'
    with open(file_name) as f:
        if (name == 'train'):
            train_json = json.load(f)
        elif (name == 'dev'):
            dev_json = json.load(f)
        elif (name == 'test'):
            test_json = json.load(f)

adaption_set = adaptionData(train_json)
train_set = rescoreDataset(train_json)
dev_set = rescoreDataset(dev_json)
test_set = rescoreDataset(test_json)


"""Training Dataloader"""
adaption_loader = DataLoader(
    dataset=adaption_set,
    batch_size=adaption_batch,
    shuffle = True,
    collate_fn=adaptionBatch
)

scoring_loader = DataLoader(
    dataset=train_set,
    batch_size=1,
    collate_fn=scoringBatch
)



dev_scoring_loader = DataLoader(
    dataset=dev_set,
    batch_size=1,
    collate_fn=scoringBatch
)

test_scoring_loader = DataLoader(
    dataset=test_set,
    batch_size=1,
    collate_fn=scoringBatch
)

train_recog_loader = DataLoader(
    dataset=train_set,
    batch_size=test_batch,
    collate_fn=scoringBatch
)
dev_loader = DataLoader(
    dataset=dev_set,
    batch_size=test_batch,
    collate_fn=scoringBatch
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=test_batch,
    collate_fn=scoringBatch
)

nBest = 10

"""Init model"""
logging.warning(f'device:{device}')
device = torch.device(device)
model = RescoreBert(
    train_batch=train_batch,
    test_batch=test_batch,
    nBest=nBest,
    use_MWER=use_MWER,
    use_MWED=use_MWED,
    device=device)
adapt_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
train_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

scoring_set = ['train', 'dev', 'test']

if (stage <= 1):
    print(f"domain adaption : {adaption_mode} mode")
    adapt_optimizer.zero_grad()
    if (adapt_epoch > 0):
        for e in range(adapt_epoch):
            model.train()
            logging_loss = 0.0
            for n, data in enumerate(tqdm(adaption_loader)):
                token, seg, mask = data
                token = token.to(device)
                seg = seg.to(device)
                mask = mask.to(device)

                loss = model.adaption(token, seg, mask, mode=adaption_mode)
                loss = loss / accumgrad
                loss.backward()
                logging_loss += loss.clone().detach().cpu()

                if (((n + 1) % accumgrad == 0) or ((n + 1) == len(adaption_loader))):
                    adapt_optimizer.step()
                    adapt_optimizer.zero_grad()

                if ((n + 1) % print_loss == 0):
                    logging.warning(
                        f'Adaption epoch :{e + 1} step:{n + 1}, adaption loss:{logging_loss}')
                    logging_loss = 0.0
            model_args['state_dict'] = model.state_dict()
            torch.save(
                model_args, f"./checkpoint/adaption/checkpoint_adapt_{e + 1}.pt")

if (stage <= 2):
    """PLL Scoring"""
    print(f'PLL scoring:')
    if (stage == 2):
        train_args = torch.load(
            f"./checkpoint/adaption/checkpoint_adapt_{adapt_epoch}.pt")
        model.load_state_dict(train_args['state_dict'])
    model.eval()
    pll_data = train_json
    pll_loader = scoring_loader
    for t in scoring_set:
        if (t == 'train'):
            pll_loader = scoring_loader
            pll_data = train_json
        elif (t == 'dev'):
            pll_loader = dev_scoring_loader
            pll_data = dev_json
        elif (t == 'test'):
            pll_loader = test_scoring_loader
            pll_data = test_json
        with torch.no_grad():
            for n, data in enumerate(tqdm(pll_loader)):
                name, token, seg, mask, _, _ = data
                
                token = token.to(device)
                seg = seg.to(device)
                mask = mask.to(device)
                    
                pll_score = model.pll_scoring(token, seg, mask)

                # train_json during training
                for i, data in enumerate(pll_data):
                    if (data['name'] == name):
                        # train_json during training
                        pll_data[i]['pll'] = pll_score.tolist()

            # debug
        for i, data in enumerate(pll_data):
            assert('pll' in data.keys()), 'PLL score not exist.'

        with open(f'./data/aishell_{t}/token_pll.json', 'w') as f:
            json.dump(pll_data, f, ensure_ascii=False, indent=2)


train_json = None
valid_json = None
with open(f'./data/aishell_train/token_pll.json', 'r') as f:
    train_json = json.load(f)
train_set = nBestDataset(train_json)
train_loader = DataLoader(
    dataset=train_set,
    batch_size=train_batch,
    collate_fn=createBatch
)

with open(f'./data/aishell_dev/token_pll.json', 'r') as f:
    valid_json = json.load(f)
valid_set = nBestDataset(valid_json)
valid_loader = DataLoader(
    dataset=valid_set,
    batch_size=test_batch,
    collate_fn=createBatch
)

"""training"""
if (stage <= 3):
    if (stage == 3):
        train_args = torch.load(
            f"./checkpoint/adaption/checkpoint_adapt_{adapt_epoch}.pt")
        model.load_state_dict(train_args['state_dict'])
    print(f'training...')
    train_optimizer.zero_grad()

    last_val = 1e8
    for e in range(epochs):
        print(f'epoch:{e + 1}')
        model.train()
        accum_loss = 0.0
        logging_loss = 0.0
        for n, data in enumerate(tqdm(train_loader)):
            # if (n < 16000):
            #     continue
            token, seg, mask, score, cer, pll = data
            token = token.to(device)
            seg = seg.to(device)
            mask = mask.to(device)
            score = score.to(device)
            cer = cer.to(device)
            pll = pll.to(device)

            loss = model(token, seg, mask, score, cer, pll)
            loss = loss / accumgrad
            loss.backward()

            logging_loss += loss.clone().detach().cpu()

            if ((n + 1) % accumgrad == 0 or (n + 1) == len(train_loader)):
                train_optimizer.step()
                train_optimizer.zero_grad()

            if ((n + 1) % print_loss == 0):
                logging.warning(
                    f'Training epoch:{e + 1} step:{n + 1}, training loss:{logging_loss / print_loss}')
                logging_loss = 0.0

        model_args['state_dict'] = model.state_dict()
        torch.save(
            model_args, f"./checkpoint/{training}/checkpoint_train_{e + 1}.pt")
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            c = 0 # correction
            i = 0 # insertion
            d = 0 # deletion
            s = 0 # substition
            print('validation')
            for n, data in enumerate(tqdm(valid_loader)):
                token, seg, mask, score, cer, pll = data
                token = token.to(device)
                seg = seg.to(device)
                mask = mask.to(device)
                score = score.to(device)
                cer = cer.to(device)
                pll = pll.to(device)

                loss, cers = model(token, seg, mask, score, cer, pll)
                val_loss += loss
                c += cers[0]
                s += cers[1]
                d += cers[2]
                i += cers[3]
            val_loss = val_loss / len(dev_loader)
            val_cer = (s + d + i) / (c + s + d)

        logging.warning(f'epoch :{e + 1}, validation_loss:{val_loss}')
        logging.warning(f'epoch :{e + 1}, validation_CER:{val_cer}\n')
        if (last_val - val_loss < 1e-4):
            print('early stop')
            logging.warning(f'early stop')
            epoch = e + 1
            break
        last_val = val_loss

find_weight = True

# recognizing
if (stage <= 4):
    print(f'recognizing')
    if (stage == 4):
        print(
            f'using checkpoint: ./checkpoint/{training}/checkpoint_train_9.pt')
        model_args = torch.load(
            f'./checkpoint/{training}/checkpoint_train_9.pt')
        model.load_state_dict(model_args['state_dict'])

    model.eval()
    recog_set = ['train', 'dev', 'test']
    recog_data = None
    with torch.no_grad():
        print(f'rescoring')
        for task in recog_set:
            print(f'recogizing: {task}')
            if (task == 'train'):
                recog_data = train_recog_loader
            elif (task == 'dev'):
                recog_data = dev_loader
            elif (task == 'test'):
                recog_data = test_loader
            recog_dict = dict()
            recog_dict['utts'] = dict()
            for n, data in enumerate(tqdm(recog_data)):
                _, token, seg, mask, score, ref = data
                token = token.to(device)
                seg = seg.to(device)
                mask = mask.to(device)
                score = score.to(device)
                rescore, _, _ , _ = model.recognize(
                        token, seg, mask, score, weight = 0)
                recog_dict['utts'][f'{task}_{n}'] = dict()
                recog_dict['utts'][f'{task}_{n}']['output'] = {
                    'token': token.tolist(),
                    "first_score": score.tolist(),
                    "second_score": rescore.tolist(),
                }
            with open(f'./data/aishell_{task}/rescore/{training}_recog_score.json', 'w') as f:
                json.dump(recog_dict, f, ensure_ascii=False, indent=2)
        
        if (find_weight):
            print(f'Finding Best weight')
            valid_json = None
            with open(f'./data/rescore/MD_recog_score.json') as f:
                valid_json = json.load()
            best_cer = 100
            best_weight = 0
            for w in tqdm(range(100)):
                correction = 0  # correction
                substitution = 0  # substitution
                deletion = 0  # deletion
                insertion = 0  # insertion

                weight = w * 0.01
                for data in valid_loader:
                    token, seg, mask, score, cer, pll = data
                    token = token.to(device)
                    seg = seg.to(device)
                    mask = mask.to(device)
                    score = score.to(device)

                    cer = cer.view(test_batch, nBest, -1)
                    _, _, _, max_index = model.recognize(token, seg, mask, score, weight = weight)

                    for i, j in enumerate(max_index.tolist()):
                        correction += cer[i][j][0]
                        substitution += cer[i][j][1]
                        deletion += cer[i][j][2]
                        insertion += cer[i][j][3]

                cer = (substitution + deletion + insertion) / \
                    (correction + substitution + deletion )
                logging.warning(f'weight:{weight}, cer:{cer}')
                if (best_cer > cer):
                    print(f'update weight:{weight}, cer:{cer}\r')
                    best_cer = cer
                    best_weight = weight
        else:
            best_weight = 0.04

        print(f'Best weight at: {best_weight}')
        for task in recog_set:
            print(f'recogizing: {task}')
            if (task == 'dev'):
                recog_data = dev_loader
            elif (task == 'test'):
                recog_data = test_loader

            recog_dict = dict()
            recog_dict['utts'] = dict()

            for i in range(100):
                # remove [CLS] and [SEP]
                token_list = [str(t) for t in best_hyp[0]]
                ref_list = [str(t) for t in ref[0][5:-5]]
                recog_dict['utts'][f'{task}_{n}'] = dict()
                recog_dict['utts'][f'{task}_{n}']['output'] = {
                    'rec_text': "".join(token_list),
                    'rec_token': " ".join(token_list),
                    "first_score": score.tolist(),
                    "second_score": rescore.tolist(),
                    "rescore": weighted_score.tolist(),
                    "text": "".join(ref_list),
                    "text_token": " ".join(ref_list)
                }

            with open(f'data/aishell_{task}/rescore/{training}_rescore_data.json', 'w') as f:
                json.dump(recog_dict, f, ensure_ascii=False, indent=2)

    print('Finish')

import os
from tqdm import tqdm
import random
import json
import yaml
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from BertForRescoring.RescoreBert import RescoreBert, MLMBert
from transformers import BertTokenizer

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

"""Basic setting"""
# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = f'./config/RescoreBert.yaml'
adapt_args = dict()
train_args = dict()
recog_args = dict()

with open(config, 'r') as f:
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    stage = conf['stage']
    adapt_args = conf['adapt']
    train_args = conf['train']
    recog_args = conf['recog']

print(f'stage:{stage}')
# adaption
adapt_epoch = adapt_args['epoch']
adapt_lr = float(adapt_args['lr'])
adapt_mode = adapt_args['mode']

if (adapt_mode == 'sequence'):
    adapt_batch = adapt_args['mlm_batch']
else:
    adapt_batch = adapt_args['train_batch']


# training
epochs = train_args['epoch']
train_batch = train_args['train_batch']
accumgrad = train_args['accumgrad']
print_loss = train_args['print_loss']
train_lr = float(train_args['lr'])

training = train_args['mode']
use_MWER = False
use_MWED = False
print(f'training mode:{training}')
if (training == 'MWER'):
    use_MWER = True
elif (training == 'MWED'):
    use_MWED = True

# recognition
recog_batch = recog_args['batch']
find_weight = recog_args['find_weight']

""""""

FORMAT = '%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s'
logging.basicConfig(level=logging.INFO,
                    filename=f'./log/{training}_train.log', filemode='w', format=FORMAT)

adapt_checkpoint = {
    'state_dict': None,
    'optimizer': None,
    'last_val_loss': None
}

train_checkpoint = {
    'training': None,
    'state_dict': None,
    'optimizer': None,
    'last_val_loss': None
}
train_checkpoint['training'] = training

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
            self.data[idx]['ref'],\
            self.data[idx]['err']

    def __len__(self):
        return len(self.data)

# For domain adaption


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

    cer = [s[5] for s in sample]

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

    return name[0], tokens, segs, masks, torch.tensor(scores), ref, cer

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
        assert (len(p) == len(s[0])), f'illegal pll:{p}'
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


train_json = None
dev_json = None
test_json = None
print(f'Prepare data')
with open(train_args['train_json']) as f, \
     open(train_args['dev_json']) as d, \
     open(train_args['test_json']) as t:
    train_json = json.load(f)
    dev_json = json.load(d)
    test_json = json.load(t)


adaption_set = adaptionData(train_json)
train_set = rescoreDataset(train_json)
dev_set = rescoreDataset(dev_json)
test_set = rescoreDataset(test_json)
num_nBest = len(train_json[0]['token'])

"""Training Dataloader"""
adaption_loader = DataLoader(
    dataset=adaption_set,
    batch_size=adapt_batch,
    shuffle=True,
    collate_fn=adaptionBatch,
)

scoring_loader = DataLoader(
    dataset = train_set,
    batch_size = recog_batch,
    collate_fn = scoringBatch,
    pin_memory = True,
    num_workers = 3
)


test_scoring_loader = DataLoader(
    dataset=test_set,
    batch_size=1,
    collate_fn=scoringBatch
)

train_recog_loader = DataLoader(
    dataset=train_set,
    batch_size=recog_batch,
    collate_fn=scoringBatch
)
dev_loader = DataLoader(
    dataset=dev_set,
    batch_size=recog_batch,
    collate_fn=scoringBatch,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=recog_batch,
    collate_fn=scoringBatch,
    pin_memory=True,
)

nBest = 10

"""Init model"""
logging.warning(f'device:{device}')
device = torch.device(device)

teacher = MLMBert(
    train_batch=adapt_batch,
    test_batch=recog_batch,
    nBest=num_nBest,
    device=device,
    mode=adapt_mode,
    lr=adapt_lr,
)

model = RescoreBert(
    train_batch=train_batch,
    test_batch=recog_batch,
    nBest=num_nBest,
    use_MWER=use_MWER,
    use_MWED=use_MWED,
    device=device,
    lr=train_lr,
    weight=0.59
)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

scoring_set = ['train', 'dev', 'test']

if (stage <= 1):
    adapt_loss = []
    print(f"domain adaption : {adapt_mode} mode")
    teacher.optimizer.zero_grad()
    if (adapt_epoch > 0):
        for e in range(adapt_epoch):
            model.train()
            logging_loss = 0.0
            for n, data in enumerate(tqdm(adaption_loader)):
                token, seg, mask = data
                token = token.to(device)
                seg = seg.to(device)
                mask = mask.to(device)

                loss = teacher(token, seg, mask)
                loss = loss / accumgrad
                loss.backward()
                logging_loss += loss.clone().detach().cpu()

                if (((n + 1) % accumgrad == 0) or ((n + 1) == len(adaption_loader))):
                    teacher.optimizer.step()
                    teacher.optimizer.zero_grad()

                if ((n + 1) % print_loss == 0):
                    logging.warning(
                        f'Adaption epoch :{e + 1} step:{n + 1}, adaption loss:{logging_loss}'
                    )
                    adapt_loss.append(logging_loss / print_loss)
                    logging_loss = 0.0
            adapt_checkpoint['state_dict'] = teacher.model.state_dict()
            adapt_checkpoint['optimizer'] = teacher.optimizer.state_dict()
            torch.save(
                adapt_checkpoint, f"./checkpoint/adaption/checkpoint_adapt_{e + 1}.pt"
            )

        if (not os.path.exists('./log/RescoreBert')):
            os.makedirs('./log/RescoreBert')
        torch.save(adapt_loss, './log/RescoreBert/adaption_loss.pt')

if (stage <= 2):
    """PLL Scoring"""
    print(f'PLL scoring:')
    if (stage == 2):
        adapt_checkpoint = torch.load(
            f"./checkpoint/adaption/checkpoint_adapt_{adapt_epoch}.pt"
        )
        teacher.model.load_state_dict(adapt_checkpoint['state_dict'])

    model.eval()
    pll_data = train_json
    pll_loader = scoring_loader
    for t in scoring_set:
        if (t == 'train'):
            pll_data = train_json
            pll_loader = scoring_loader

        elif (t == 'dev'):
            pll_data = dev_json
            pll_loader = dev_loader

        elif (t == 'test'):
            pll_data = test_json
            pll_loader = test_loader

        with torch.no_grad():
            for n, data in enumerate(tqdm(pll_loader)):
                name, token, seg, mask, _, _, _ = data

                token = token.to(device)
                seg = seg.to(device)
                mask = mask.to(device)

                pll_score = teacher.recognize(token, seg, mask)

                # train_json during training
                for i, data in enumerate(pll_data):
                    if (data['name'] == name):
                        # train_json during training
                        pll_data[i]['pll'] = pll_score.tolist()

            # debug
        for i, data in enumerate(pll_data):
            assert('pll' in data.keys()), 'PLL score not exist.'

        with open(f'./data/aishell_{t}/token_pll.json', 'w') as f:
            json.dump(pll_data, f, ensure_ascii=False, indent=4)


train_json = None
valid_json = None
with open(f'./data/aishell_train/token_pll.json', 'r') as f:
    train_json = json.load(f)
train_set = nBestDataset(train_json)
train_loader = DataLoader(
    dataset=train_set,
    batch_size=train_batch,
    collate_fn=createBatch,
    pin_memory=True,
    shuffle=True,
    num_workers=3
)

with open(f'./data/aishell_dev/token_pll.json', 'r') as f:
    valid_json = json.load(f)
valid_set = nBestDataset(valid_json)
valid_loader = DataLoader(
    dataset=valid_set,
    batch_size=recog_batch,
    collate_fn=createBatch
)

"""training"""
if (stage <= 3):
    print(f'training...')
    model.optimizer.zero_grad()

    last_val = 1e8
    train_loss = []
    dev_loss = []
    dev_cers = []
    for e in range(epochs):
        model.train()
        accum_loss = 0.0
        logging_loss = 0.0
        for n, data in enumerate(tqdm(train_loader)):
            token, seg, mask, score, cer, pll = data
            token = token.to(device)
            seg = seg.to(device)
            mask = mask.to(device)
            score = score.to(device)
            cer = cer.to(device)
            pll = pll.to(device)

            loss = model(token, seg, mask, score, cer, pll)
            loss = loss / accumgrad
            logging.warning(f'loss:{loss}')
            loss.backward()

            logging_loss += loss.clone().detach().cpu()

            if ((n + 1) % accumgrad == 0 or (n + 1) == len(train_loader)):
                model.optimizer.step()
                model.optimizer.zero_grad()

            if ((n + 1) % print_loss == 0 or (n + 1) == len(train_loader)):
                train_loss.append(logging_loss / print_loss)
                logging.warning(
                    f'Training epoch:{e + 1} step:{n + 1}, training loss:{logging_loss / print_loss}')
                logging_loss = 0.0

        train_checkpoint['state_dict'] = model.model.state_dict()
        train_checkpoint['optimizer'] = model.optimizer.state_dict()
        torch.save(
            train_checkpoint, f"./checkpoint/{training}/checkpoint_train_{e + 1}.pt"
        )

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_cer = 0.0
            c = 0
            s = 0
            d = 0
            i = 0
            for n, data in enumerate(tqdm(valid_loader)):
                token, seg, mask, score, cer, pll = data
                token = token.to(device)
                seg = seg.to(device)
                mask = mask.to(device)
                score = score.to(device)
                cer = cer.to(device)
                pll = pll.to(device)

                loss, err = model(token, seg, mask, score, cer, pll)
                val_loss += loss
                c += err[0]
                s += err[1]
                d += err[2]
                i += err[3]

            val_cer = (s + d + i) / (c + s + d)
            val_loss = val_loss / len(dev_loader)
            dev_loss.append(val_loss)
            dev_cers.append(val_cer)

        logging.warning(f'epoch :{e + 1}, validation_loss:{val_loss}')
        logging.warning(f'epoch :{e + 1}, validation_loss:{val_cer}')

        if (last_val - val_loss < 1e-4):
            print('early stop')
            logging.warning(f'early stop')
            epochs = e + 1
            break
        last_val = val_loss

    logging_loss = {
        'training_loss': train_loss,
        'dev_loss': dev_loss,
        'dev_cer': dev_cers
    }
    if (not os.path.exists(f'./log/RescoreBert')):
        os.makedirs('./log/RescoreBert')
    torch.save(logging_loss, f'./log/RescoreBert/loss.pt')

# recognizing
if (stage <= 4):
    print(f'scoring')
    if (stage == 4):
        print(
            f'using checkpoint: ./checkpoint/{training}/checkpoint_train_{epochs}.pt'
        )
        checkpoint = torch.load(
            f'./checkpoint/{training}/checkpoint_train_{epochs}.pt'
        )
        model.model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
    recog_set = ['train', 'dev', 'test']
    recog_data = None
    with torch.no_grad():
        for task in recog_set:
            print(f'scoring: {task}')
            if (task == 'train'):
                recog_data = scoring_loader
            elif (task == 'dev'):
                recog_data = dev_loader
            elif (task == 'test'):
                recog_data = test_loader

            recog_dict = []
            for n, data in enumerate(tqdm(recog_data)):
                _, token, seg, mask, score, ref, cer = data
                token = token.to(device)
                seg = seg.to(device)
                mask = mask.to(device)
                score = score.to(device)

                rescore, _, _, _ = model.recognize(
                    token, seg, mask, score, weight=1
                )

                recog_dict.append(
                    {
                        'token': token.tolist(),
                        'ref': ref,
                        'cer': cer,
                        'first_score': score.tolist(),
                        'rescore': rescore.tolist()
                        }
                )
            with open(f'data/aishell_{task}/rescore/{training}_recog_data.json', 'w') as f:
                json.dump(recog_dict, f, ensure_ascii=False, indent=4)

if (stage <= 5):
    # find best weight
    if (find_weight):
        print(f'Finding Best weight')
        val_score = None
        with open(f'data/aishell_dev/rescore/{training}_recog_data.json') as f:
            val_score = json.load(f)

        best_cer = 100
        best_weight = 0
        for w in tqdm(range(100)):
            correction = 0  # correction
            substitution = 0  # substitution
            deletion = 0  # deletion
            insertion = 0  # insertion

            weight = w * 0.01
            for data in val_score:
                first_score = torch.tensor(data['first_score'])
                rescore = torch.tensor(data['rescore'])
                cer = torch.tensor(data['cer'])
                cer = cer.view(-1, 4)

                weighted_score = first_score + weight*rescore
                
                max_index = torch.argmax(weighted_score)
                
                correction += cer[max_index][0]
                substitution += cer[max_index][1]
                deletion += cer[max_index][2]
                insertion += cer[max_index][3]

            cer = (substitution + deletion + insertion) / \
                (correction + deletion + substitution)
            logging.warning(f'weight:{weight}, cer:{cer}')
            if (best_cer > cer):
                print(f'update weight:{weight}, cer:{cer}\r')
                best_cer = cer
                best_weight = weight
    else:
        best_weight = 0.59

if (stage <= 6):
    print('output result')
    if (stage == 6):
        best_weight = 0.59
    print(f'Best weight at: {best_weight}')
    recog_set = ['train', 'dev', 'test']
    for task in recog_set:
        print(f'recogizing: {task}')
        score_data = None
        with open(f'data/aishell_{task}/rescore/{training}_recog_data.json') as f:
            score_data = json.load(f)

        recog_dict = dict()
        recog_dict['utts'] = dict()
        for n, data in enumerate(score_data):
            token = data['token']
            ref = data['ref']

            score = torch.tensor(data['first_score'])
            rescore = torch.tensor(data['rescore'])

            weight_sum = score + best_weight * rescore

            max_index = torch.argmax(weight_sum).item()

            best_hyp = token[max_index]
           
            sep = best_hyp.index(102)
            best_hyp = tokenizer.convert_ids_to_tokens(t for t in best_hyp[1:sep])
            ref = list(ref[0][5:-5])
            # remove [CLS] and [SEP]
            token_list = [str(t) for t in best_hyp]
            ref_list = [str(t) for t in ref]
            recog_dict['utts'][f'{task}_{n + 1}'] = dict()
            recog_dict['utts'][f'{task}_{n + 1}']['output'] = {
                    'rec_text': "".join(token_list),
                    'rec_token': " ".join(token_list),
                    "first_score": score.tolist(),
                    "second_score": rescore.tolist(),
                    "rescore": weight_sum.tolist(),
                    "text": "".join(ref_list),
                    "text_token": " ".join(ref_list)
            }

        with open(f'data/aishell_{task}/rescore/{training}_rescore_data.json', 'w') as f:
            json.dump(recog_dict, f, ensure_ascii=False, indent=4)

print('Finish')

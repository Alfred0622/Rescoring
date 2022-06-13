import os
from tqdm import tqdm
import random
import yaml
import logging
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from BertForRescoring.BartForCorrection import RoBart
from transformers import BertTokenizer

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

config = f"./config/RoBart.yaml"

adapt_args = dict()
train_args = dict()
recog_args = dict()

with open(config, "r") as f:
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    stage = conf["stage"]
    stop_stage = conf["stop_stage"]
    train_args = conf["train"]
    recog_args = conf["recog"]

epochs = train_args["epoch"]
train_batch = train_args["train_batch"]
accumgrad = train_args["accumgrad"]
print_loss = train_args["print_loss"]
train_lr = float(train_args["lr"])

recog_batch = recog_args["batch"]
find_weight = recog_args["find_weight"]

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/RoBart/train.log",
    filemode="w",
    format=FORMAT,
)


class correctDataset(Dataset):
    def __init__(self, nbest_list):
        self.data = nbest_list

    def __getitem__(self, idx):
        return (
            self.data[idx]["token"],
            self.data[idx]["phoneme"],
            self.data[idx]["ref_token"],
        )

    def __len__(self):
        return len(self.data)


def createBatch(sample):
    token_id = [s[0] + s[1] for s in sample]
    seg_id = [[0 * len(s[0])] + [1 * len(s[1])] for s in sample]

    for i, token in enumerate(token_id):
        token_id[i] = torch.tensor(token)
        seg_id[i] = torch.tensor(seg_id[i])

    token_id = pad_sequence(token_id, batch_first=True)
    seg_id = pad_sequence(seg_id, batch_first=True, padding_value=1)

    attention_mask = torch.zeros(token_id.shape)
    attention_mask = attention_mask.masked_fill(token_id != 0, 1)

    labels = [s[2] for s in sample]

    for i, label in labels:
        labels[i] = torch.tensor(label)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return token_id, seg_id, attention_mask, labels


train_json = None
dev_json = None
test_json = None
print(f"Prepare data")
with open(train_args["train_json"]) as f, open(train_args["dev_json"]) as d, open(
    train_args["test_json"]
) as t:
    train_json = json.load(f)
    dev_json = json.load(d)
    test_json = json.load(t)

train_set = correctDataset(train_json)
dev_set = correctDataset(dev_json)
test_set = correctDataset(test_json)

train_loader = DataLoader(
    dataset=train_set,
    batch_size=train_batch,
    collate_fn=createBatch,
    pin_memory=True,
    num_workers=4,
)

dev_loader = DataLoader(
    dataset=dev_set,
    batch_size=recog_batch,
    collate_fn=createBatch,
    pin_memory=True,
    num_workers=4,
)

test_loader = DataLoader(
    dataset=test_set,
    batch_size=recog_batch,
    collate_fn=createBatch,
    pin_memory=True,
    num_workers=4,
)

device = torch.device(device)
model = RoBart(device)

if stage >= 0:
    print("training")
    last_val = 1e8
    dev_loss = []
    train_loss = []
    for e in epochs:
        for n, data in enumerate(train_loader):
            token, seg, mask, label = data
            token = token.to(device)
            seg = seg.to(device)
            mask = mask.to(device)
            label = label.to(device)

            loss = model(token, seg, mask, label)

            loss /= accumgrad
            loss.backward()
            logging_loss += loss.clone().detach().cpu()

            if ((n + 1) % accumgrad == 0) or ((n + 1) == len(train_loader)):
                model.optimizer.step()
                model.optimizer.zero_grad()

            if (n + 1) % print_loss == 0 or (n + 1) == len(train_loader):
                logging.warning(
                    f"Training epoch :{e + 1} step:{n + 1}, training loss:{logging_loss}"
                )
                train_loss.append(logging_loss / print_loss)
                logging_loss = 0.0

            # train_checkpoint["epoch"] = epochs + 1
            # train_checkpoint["state_dict"] = model.model.state_dict()
            # train_checkpoint["optimizer"] = model.optimizer.state_dict()
            # torch.save(
            #     train_checkpoint,
            #     f"./checkpoint/nBestTransformer/{training_mode}/{model_name}/checkpoint_train_{e + 1}.pt",
            # )

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for n, data in enumerate(tqdm(dev_loader)):
                    token, mask, label, label_text = data
                    token = token.to(device)
                    mask = mask.to(device)
                    label = label.to(device)

                    loss = model(token, mask, label)
                    val_loss += loss

                val_loss = val_loss / len(dev_loader)
                dev_loss.append(val_loss)

                logging.warning(f"epoch :{e + 1}, validation_loss:{val_loss}")

                if val_loss < min_val:
                    min_val = val_loss
                    min_epoch = e
                    stage = 2

        logging_loss = {
            "training_loss": train_loss,
            "dev_loss": dev_loss,
        }
        if not os.path.exists(f"./log/RoBart"):
            os.makedirs("./log/RoBart")
        torch.save(logging_loss, f"./log/RoBart/loss.pt")

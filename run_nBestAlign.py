import os
from tqdm import tqdm
import random
import json
import yaml
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nBestAligner.nBestTransformer import nBestTransformer
from transformers import BertTokenizer

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

config = f"./config/nBestAlign.yaml"
train_args = dict()
recog_args = dict()

with open(config, "r") as f:
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    stage = conf["stage"]
    train_args = conf["train"]
    recog_args = conf["recog"]

epochs = train_args["epoch"]
train_batch = train_args["train_batch"]
accumgrad = train_args["accumgrad"]
print_loss = train_args["print_loss"]
train_lr = float(train_args["lr"])

training_mode = train_args["mode"]

print(f"training mode:{training_mode}")


# recognition
recog_batch = recog_args["batch"]

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/nBestTransformer/{training_mode}_train.log",
    filemode="w",
    format=FORMAT,
)

train_checkpoint = {
    "training": None,
    "state_dict": None,
    "optimizer": None,
    "last_val_loss": None,
}


class rescoreDataset(Dataset):
    def __init__(self, nbest_list):
        """
        nbest_dict: {token seq : CER}
        """
        self.data = nbest_list

    def __getitem__(self, idx):
        return (
            self.data[idx]["token"],
            self.data[idx]["ref_token"],
            self.data[idx]["ref"],
        )

    def __len__(self):
        return len(self.data)


def createBatch(sample):
    tokens = [s[0] for s in sample]

    ref_tokens = [s[1] for s in sample]

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)

    for i, t in enumerate(ref_tokens):
        ref_tokens[i] = torch.tensor(t)

    tokens = pad_sequence(tokens, batch_first=True)
    ref_tokens = pad_sequence(ref_tokens, batch_first=True)

    masks = torch.zeros(tokens.shape[:2], dtype=torch.long)
    
    masks = masks.masked_fill(torch.all(tokens != torch.zeros(tokens.shape[-1])), 1)

    ref = [s[2] for s in sample]

    return tokens, masks, ref_tokens, ref


if __name__ == "__main__":
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

    train_set = rescoreDataset(train_json)
    dev_set = rescoreDataset(dev_json)
    test_set = rescoreDataset(test_json)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_batch,
        collate_fn=createBatch,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    dev_loader = DataLoader(
        dataset=dev_set,
        batch_size=recog_batch,
        collate_fn=createBatch,
        shuffle=True,
        pin_memory=True,
        num_workers=3,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=recog_batch,
        collate_fn=createBatch,
        shuffle=True,
        pin_memory=True,
        num_workers=3,
    )

    nBest = len(train_json[0]["token"][0])

    if stage <= 1:
        logging.warning(f"device:{device}")
        device = torch.device(device)

        model = nBestTransformer(
            nBest=nBest,
            train_batch=train_batch,
            test_batch=recog_batch,
            device=device,
            lr=train_lr,
            mode=training_mode,
        )

        # scoring_set = ["train", "dev", "test"]

        dev_loss = []
        train_loss = []
        last_val = 1e8
        for e in range(epochs):
            model.train()

            logging_loss = 0.0
            for n, data in enumerate(tqdm(train_loader)):
                token, mask, label, _ = data
                token = token.to(device)
                mask = mask.to(device)
                label = label.to(device)

                loss = model(token, mask, label)

                loss /= accumgrad
                loss.backward()
                logging_loss += loss.clone().detach().cpu()

                if ((n + 1) % accumgrad == 0) or ((n + 1) == len(train_loader)):
                    model.optimizer.step()
                    model.optimizer.zero_grad()

                if (n + 1) % print_loss == 0:
                    logging.warning(
                        f"Adaption epoch :{e + 1} step:{n + 1}, adaption loss:{logging_loss}"
                    )
                    train_loss.append(logging_loss / print_loss)
                    logging_loss = 0.0
                train_checkpoint["state_dict"] = model.model.state_dict()
                train_checkpoint["optimizer"] = model.optimizer.state_dict()
                torch.save(
                    train_checkpoint,
                    f"./checkpoint/nBestTransformer/{training_mode}/checkpoint_train_{e + 1}.pt",
                )

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for n, data in enumerate(tqdm(dev_loader)):
                    token, mask, label, _ = data
                    token = token.to(device)
                    mask = mask.to(device)
                    label = label.to(device)

                    loss = model(token, mask, label)
                    val_loss += loss

                val_loss = val_loss / len(dev_loader)
                dev_loss.append(val_loss)

                logging.warning(f"epoch :{e + 1}, validation_loss:{val_loss}")

                if last_val - val_loss < 1e-4:
                    print("early stop")
                    logging.warning(f"early stop")
                    epochs = e + 1
                    break
                last_val = val_loss
    logging_loss = {
        "training_loss": train_loss,
        "dev_loss": dev_loss,
    }
    if not os.path.exists(f"./log/RescoreBert/nBestTransformer"):
        os.makedirs("./log/RescoreBert/nBestTransformer")
    torch.save(logging_loss, f"./log/RescoreBert/nBestTransformer/loss.pt")

    if stage <= 2:
        print("recognizing")
        recog_set = ["dev", "test"]
        for task in recog_set:
            if task == "dev":
                recog_loader = dev_loader
            elif task == "test":
                recog_loader = test_loader
            recog_dict = dict()
            for n, data in enumerate(recog_loader):
                name = f"{task}_{n}"
                token, mask, _, ref_text = data
                token = token.to(device)
                mask = mask.to(device)

                output = model.recognize(token, mask)

                logging.warning(output)

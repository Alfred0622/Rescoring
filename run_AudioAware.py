from ast import AugAssign
import os
from tqdm import tqdm
import random
import json
import yaml
import logging
import torch
from chainer.datasets import TransformDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from models.AudioAware.AudioAwareReranker import AudioAwareReranker
import kaldiio

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
config = f"./config/AudioAware.yaml"

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
# find_weight = recog_args["find_weight"]

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/Audio_Aware/Audio_Aware.log",
    filemode="w",
    format=FORMAT,
)

# Define Dataset
class AudioDataset(Dataset):
    def __init__(self, nbest_list, nbest):
        self.data = nbest_list
        self.audio_feat = [
            torch.from_numpy(kaldiio.load_mat(data["feat"])) for data in self.data
        ]

        self.nbest = nbest

    def __getitem__(self, idx):
        return (
            self.audio_feat[idx],
            self.data[idx]["nbest_token"][: self.nbest],
            self.data[idx]["ref_token"][: self.nbest],
            self.data[idx]["nbest"][: self.nbest],
            self.data[idx]["ref"],
        )

    def __len__(self):
        return len(self.data)


def createBatch(sample):
    token_id = []
    labels = []
    audio_feat = []
    audio_lens = []
    for s in sample:
        audio_feat += [s[0] for _ in range(3)]
        audio_lens += [s[0].shape[0] for _ in range(3)]
        token_id += s[1]
        labels += s[2]

    audio_lens = torch.tensor(audio_lens)

    for i, (token, label) in enumerate(zip(token_id, labels)):
        token_id[i] = torch.tensor(token)
        labels[i] = torch.tensor(label)

    audio_feat = pad_sequence(audio_feat, batch_first=True)
    token_id = pad_sequence(token_id, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    # attention_mask = pad_sequence(attention_mask, batch_first=True)
    masks = torch.zeros(token_id.shape, dtype=torch.long)
    masks = masks.masked_fill(token_id != 0, 1)

    texts = [s[3] for s in sample]

    ref = s[4]

    return audio_feat, audio_lens, token_id, masks, labels, texts, ref


train_json = None
dev_json = None
test_json = None
debug_model = True
print(f"Prepare data")
logging.warning("Prepare data")
with open(train_args["train_json"]) as f, open(train_args["dev_json"]) as d, open(
    train_args["test_json"]
) as t:
    train_json = json.load(f)
    dev_json = json.load(d)
    test_json = json.load(t)

if debug_model == True:
    train_set = AudioDataset(train_json[:train_batch], 3)
    dev_set = AudioDataset(dev_json[:recog_batch], 3)
    test_set = AudioDataset(test_json[:recog_batch], 3)
else:
    train_set = AudioDataset(train_json, 3)
    dev_set = AudioDataset(dev_json, 3)
    test_set = AudioDataset(test_json, 3)

print(f"Prepare Loader")
logging.warning("Prepare Loader")
train_loader = DataLoader(
    dataset=train_set,
    batch_size=train_batch,
    collate_fn=createBatch,
    pin_memory=True,
)

dev_loader = DataLoader(
    dataset=dev_set,
    batch_size=recog_batch,
    collate_fn=createBatch,
    pin_memory=True,
)

test_loader = DataLoader(
    dataset=test_set,
    batch_size=recog_batch,
    collate_fn=createBatch,
    pin_memory=True,
)


print(f"prepare_model")
device = torch.device(device)
model = AudioAwareReranker(device)
train_checkpoint = dict()

if stage >= 0:
    print("training")
    min_val = 1e8

    dev_loss = []
    train_loss = []
    for e in range(epochs):
        print(f"epochs:{e}")
        logging_loss = 0.0
        model.train()
        for n, data in enumerate(tqdm(train_loader)):
            audio, ilens, token, mask, label, _, _ = data

            audio = audio.to(device)
            ilens = ilens.to(device)
            token = token.to(device)
            mask = mask.to(device)
            label = label.to(device)
            logging.warning(f"audio:{audio.shape}")
            loss = model(audio, ilens, token, mask, label)

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

        train_checkpoint["epoch"] = epochs + 1
        train_checkpoint["state_dict"] = model.state_dict()
        train_checkpoint["optimizer"] = model.optimizer.state_dict()
        if not os.path.exists(f"./checkpoint/Audio_Aware"):
            os.makedirs("./checkpoint/Audio_Aware")
        torch.save(
            train_checkpoint,
            f"./checkpoint/Audio_Aware/checkpoint_train_{e + 1}.pt",
        )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for n, data in enumerate(tqdm(dev_loader)):
                audio, ilens, token, mask, label, _, _ = data

                audio = audio.to(device)
                token = token.to(device)
                mask = mask.to(device)
                label = label.to(device)

                loss = model(audio, ilens, token, mask, label)
                val_loss += loss

            val_loss = val_loss / len(dev_loader)
            dev_loss.append(val_loss)

            logging.warning(f"epoch :{e + 1}, validation_loss:{val_loss}")

            if val_loss < min_val:
                min_val = val_loss
                min_epoch = e + 1
                stage = 1

        logging_loss = {
            "training_loss": train_loss,
            "dev_loss": dev_loss,
        }
        if not os.path.exists(f"./log/Audio_Aware"):
            os.makedirs("./log/Audio_Aware")
        torch.save(logging_loss, f"./log/Audio_Aware/loss.pt")

if stage <= 1:
    print("recognizing")
    if stage == 1:
        print(
            f"using checkpoint: ./checkpoint/Audio_Aware/checkpoint_train_{min_epoch}.pt"
        )
    checkpoint = torch.load(f"./checkpoint/Audio_Aware/checkpoint_train_{min_epoch}.pt")
    model.load_state_dict(checkpoint["state_dict"])
    recog_set = ["dev", "test"]

    for task in recog_set:
        if task == "dev":
            print("dev")
            recog_loader = dev_loader
        elif task == "test":
            print("test")
            recog_loader = test_loader
        elif task == "train":
            print("train")
            recog_loader = train_loader
        recog_dict = dict()
        recog_dict["utts"] = dict()
        model.eval()
        with torch.no_grad():
            for n, data in enumerate(tqdm(recog_loader)):
                name = f"{task}_{n}"
                audio, ilens, token, mask, _, texts, ref = data
                audio = audio.to(device)
                token = token.to(device)
                mask = mask.to(device)

                output = model.recognize(audio, ilens, token, mask)
                max_index = torch.argmax(output).item()

                best_hyp_token = token[max_index].tolist()
                best_hyp = list(texts[0][max_index])

                ref = [r for r in ref]

                recog_dict["utts"][name] = {
                    "rec_text": " ".join(best_hyp),
                    "ref_text": " ".join(ref),
                }

        if not os.path.exists(f"./data/aishell_{task}/50_best/Audio_Aware"):
            os.makedirs(f"./data/aishell_{task}/50_best/Audio_Aware")
        with open(
            f"./data/aishell_{task}/50_best/Audio_Aware/rescore_data.json",
            "w",
        ) as f:
            json.dump(recog_dict, f, ensure_ascii=False, indent=4)

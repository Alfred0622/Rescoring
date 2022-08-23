import os
from tqdm import tqdm
import random
import json
import yaml
import logging
import torch

from models.nBestAligner.nBestTransformer import nBestTransformer
from transformers import BertTokenizer
from utils.LoadConfig import load_config
from utils.Datasets import nBestAlignDataset
from utils.collate_fn import nBestAlignBatch

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

config = f"./config/nBestAlign.yaml"
train_args = dict()
recog_args = dict()

args, train_args, recog_args = load_config(config)

align_embedding = train_args["align_embedding"]

training_mode = train_args["mode"]
model_name = train_args["model_name"]

print(f"training mode:{training_mode}")
print(f"model name:{model_name}")

if (not os.path.exists(f"./log/nBestAlign/{training_mode}_{model_name}_train.log")):
    os.makedirs(f"./log/nBestAlign/{training_mode}_{model_name}_train.log")

FORMAT = "%(asctime)s :: %(filename)s (%(lineno)d) %(levelname)s : %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=f"./log/nBestAlign/{training_mode}_{model_name}_train.log",
    filemode="w",
    format=FORMAT,
)

train_checkpoint = {
    "training": None,
    "state_dict": None,
    "optimizer": None,
    "last_val_loss": None,
}

if __name__ == "__main__":
    train_json = None
    dev_json = None
    test_json = None
    print(f"Prepare data")
    with open(train_args["train_json"]) as f,\
         open(train_args["dev_json"]) as d, \
         open(train_args["test_json"]) as t:
        train_json = json.load(f)
        dev_json = json.load(d)
        test_json = json.load(t)

    train_set = nBestAlignDataset(train_json)
    dev_set = nBestAlignDataset(dev_json)
    test_set = nBestAlignDataset(test_json)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_args["train_batch"],
        collate_fn=nBestAlignBatch,
        pin_memory=True,
        num_workers=4,
    )

    dev_loader = DataLoader(
        dataset=dev_set,
        batch_size=recog_args['batch'],
        collate_fn=nBestAlignBatch,
        pin_memory=True,
        num_workers=4,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=recog_args['batch'],
        collate_fn=nBestAlignBatch,
        pin_memory=True,
        num_workers=4,
    )

    nBest = len(train_json[0]["token"][0])
    print(f"nBest:{nBest}")

    logging.warning(f"device:{device}")
    device = torch.device(device)

    model = nBestTransformer(
        nBest=nBest,
        train_batch=train_args["train_batch"],
        test_batch=recog_args['batch'],
        device=device,
        lr=float(train_args["lr"]),
        mode=training_mode,
        model_name=model_name,
        align_embedding=align_embedding,
    )

    min_epoch = epochs # record best epoch
    if stage <= 1:
        if (train_args['start_epoch'] > 0):
            try: # load checkpoint
                checkpoint_path = f"./checkpoint/nBestAlign/checkpoint_train_{start_epoch}.pt"
                checkpoint = torch.load(checkpoint_path)
                model.model.load_state_dict(checkpoint["state_dict"])
                model.optimizer.load_state_dict(checkpoint["optimizer"])
                start_epoch = train_args['start_epoch'] - 1
            except:
                print(f'no existing checkpoint at {train_args['start_epoch']} epoch, start from zero epoch')
                start_epoch = 0
        print(f"training")

        dev_loss = []
        train_loss = []
        min_val = 1e8

        for e in range(start_epoch, train_args['epoch']):
            model.train()

            logging_loss = 0.0
            for n, data in enumerate(tqdm(train_loader)):
                token, mask, label, label_text = data
                token = token.to(device)
                mask = mask.to(device)
                label = label.to(device)

                loss = model(token, mask, label)

                loss /= train_args["accumgrad"]
                loss.backward()
                logging_loss += loss.clone().detach().cpu()

                if ((n + 1) % train_args["accumgrad"] == 0) or ((n + 1) == len(train_loader)):
                    model.optimizer.step()
                    model.optimizer.zero_grad()

                if (n + 1) % train_args["print_loss"] == 0 or (n + 1) == len(train_loader):
                    logging.warning(
                        f"Training epoch :{e + 1} step:{n + 1}, training loss:{logging_loss}"
                    )
                    train_loss.append(logging_loss / train_args["print_loss"])
                    logging_loss = 0.0

            train_checkpoint["epoch"] = epochs + 1
            train_checkpoint["state_dict"] = model.model.state_dict()
            train_checkpoint["optimizer"] = model.optimizer.state_dict()
            torch.save(
                train_checkpoint,
                f"./checkpoint/nBestTransformer/{training_mode}/{model_name}/checkpoint_train_{e + 1}.pt",
            )

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
                    torch.save(
                        train_checkpoint,
                        f"./checkpoint/nBestTransformer/{training_mode}/{model_name}/checkpoint_train_best.pt",
                    )

        logging_loss = {
            "training_loss": train_loss,
            "dev_loss": dev_loss,
        }
        if not os.path.exists(f"./log/RescoreBert/nBestTransformer"):
            os.makedirs("./log/RescoreBert/nBestTransformer")
        torch.save(logging_loss, f"./log/RescoreBert/nBestTransformer/loss.pt")

    if stage <= 2:
        print("recognizing")
           
        checkpoint = torch.load(
            f"./checkpoint/nBestTransformer/{training_mode}/{model_name}/checkpoint_train_best.pt"
        )
        model.model.load_state_dict(checkpoint["state_dict"])
        recog_set = ["dev", "test"]
        decoder_seq = model.tokenizer.convert_tokens_to_ids(["[CLS]"])
        decoder_ids = torch.tensor(decoder_seq).unsqueeze(0).to(device)

        for task in recog_set:
            if task == "dev":
                print("dev")
                recog_loader = dev_loader
            elif task == "test":
                print("test")
                recog_loader = test_loader
            if task == "train":
                print("train")
                recog_loader = train_score_loader
            recog_dict = dict()
            recog_dict["utts"] = dict()
            model.eval()
            with torch.no_grad():
                for n, data in enumerate(tqdm(recog_loader)):
                    name = f"{task}_{n}"
                    token, mask, _, ref_text = data
                    token_list = token.squeeze(0).tolist()
                    token = token.to(device)
                    mask = mask.to(device)

                    output = model.recognize(token, mask, decoder_ids, recog_args["max_len"])
                    output = output.squeeze(0).tolist()
                    hyp_token = model.tokenizer.convert_ids_to_tokens(output)
                    hyp_token = [
                        x for x in hyp_token if x not in ["[CLS]", "[SEP]", "[PAD]"]
                    ]

                    ref_token = ref_text[0][5:-5]
                    ref_token = [str(x) for x in ref_token]

                    recog_dict["utts"][name] = {
                        "rec_text": "".join(hyp_token),
                        "rec_token": " ".join(hyp_token),
                        "text": "".join(ref_token),
                        "text_token": " ".join(ref_token),
                    }

            with open(
                f"data/aishell_{task}/10_best/{model_name}_token/rescore_data.json",
                "w",
            ) as f:
                json.dump(recog_dict, f, ensure_ascii=False, indent=4)

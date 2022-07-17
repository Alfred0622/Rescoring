import json
import yaml
import random
import torch
from torch.utils.data import Dataset, DataLoader
from models.ComparisonRescoring.BertForComparison import BertForComparison
from utils.Datasets import(
    nBestDataset,
    rescoreDataset,
    concatDataset
)
from utils.CollateFunc import(
    bertCompareBatch,
    bertCompareRecogBatch,
)

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

"""Basic setting"""
# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

config = f"./config/comparison.yaml"
adapt_args = dict()
train_args = dict()
recog_args = dict()

with open(config, "r") as f:
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    stage = conf["stage"]
    nbest = conf["nbest"]
    stop_stage = conf["stop_stage"]
    train_args = conf["train"]
    recog_args = conf["recog"]

print(f"stage:{stage}, stop_stage:{stop_stage}")

# training
epochs = train_args["epoch"]
train_batch = train_args["train_batch"]
accumgrad = train_args["accumgrad"]
print_loss = train_args["print_loss"]
train_lr = float(train_args["lr"])

# recognition
recog_batch = recog_args["batch"]
find_weight = recog_args["find_weight"]


# Prepare Data
print('Data Prepare')
train_file = train_args['train_json']
dev_file = train_args['dev_json']
test_file = train_args['test_json']
with open(train_file, 'r') as train, \
     open(dev_file, 'r') as dev, \
     open(test_file, 'r') as test:
     train_json = json.load(train)
     dev_json = json.load(dev)
     test_json = json.load(test)

train_dataset = concatDataset(train_json, nbest = 50)

dev_dataset = rescoreDataset(dev_json, nbest = 50)
test_dataset = rescoreDataset(test_json, nbest = 50)

train_loader = DataLoader(
    train_dataset,
    batch_size = train_batch,
    collate_fn=bertCompareBatch,
    pin_memory=True,
    num_workers=4,
)


dev_loader = DataLoader(
    dev_dataset,
    batch_size = recog_batch,
    collate_fn=bertCompareRecogBatch,
    pin_memory=True,
    num_workers=4,
)

test_loader = DataLoader(
    test_dataset,
    batch_size = recog_batch,
    collate_fn=bertCompareRecogBatch,
    pin_memory=True,
    num_workers=4,
)


if (stage <= 0) and (stop_stage>= 0):
    model = BertForComparision.to(device)
    print(f'training')
    min_loss = 1e8
    loss_seq = []
    for e in range(epochs):
        for n, data in enumerate(tqdm(train_loader)):
            logging_loss = 0.0
            model.train()
            token, seg, masks, labels = data
            token = token.to(device)
            seg = seg.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            loss = model(token, seg, masks, labels)
            loss = loss / accumgrad
            loss.backward()
            
            logging_loss += loss.clone().detach().cpu()

            if ((n + 1) % accumgrad == 0) or ((n + 1) == len(train_loader)):
                teacher.optimizer.step()
                teacher.optimizer.zero_grad()

            if (n + 1) % print_loss == 0:
                logging.warning(
                    f"Training epoch :{e + 1} step:{n + 1}, loss:{logging_loss}"
                )
                loss_seq.append(logging_loss / print_loss)
                logging_loss = 0.0
        
        train_checkpoint["state_dict"] = model.model.state_dict()
        train_checkpoint["optimizer"] = model.optimizer.state_dict()
        if (not os.path.exists(f'./checkpoint/Comparision/BERT/{setting}')):
            os.makedirs(f'./checkpoint/Comparision/BERT/{setting}')
        
        torch.save(
            train_checkpoint,
            f"./checkpoint/Comparision/BERT/{setting}/checkpoint_train_{e + 1}.pt",
        )

        # eval
        model.eval()
        valid_loss = 0.0
        for n, data in enumerate(tqdm(valid_loader)):
            tokens, segs, masks, _, _, _, _, _, labels = data
            loss = model(tokens, segs, masks, labels)

            valid_loss += loss
        
        if (valid_loss < min_loss):
            torch.save(
                train_checkpoint,
                f"./checkpoint/Comparision/BERT/{setting}/checkpoint_train_best.pt",
            )

            min_loss = valid_loss

if (stage <= 1) and (stop_stage >= 1):
    recog_set = ['dev', 'test']
    print(f'scoring')

    for task in recog_set:
        if (task == 'dev'):
            score_loader = dev_loader
        elif (task == 'test'):
            score_loader = test_loader
    
        recog_dict = []
        for n, data in enumerate(tqdm(score_loader)):
            tokens, segs, masks, first_score, errs, pairs, scores, texts = data
            output = model.recognize(tokens, segs, masks)
        
            for i, pair in enumerate(pairs):
                scores[pair[0]] += output[i][0]
                scores[pair[1]] += output[i][1]
            
            recog_dict.append(
                {
                    "token": tokens.tolist(),
                    "ref": texts,
                    "cer": errs,
                    "first_score": first_score.tolist(),
                    "rescore": scores.tolist(),
                }
            )

        if (not os.path.exists(f'data/aishell/{task}/BertCompare/{setting}')):
                os.makedirs(f'data/aishell/{task}/BertCompare/{setting}')
    
        print(f"writing file: ./data/aishell/{task}/BertCompare/{setting}/{nbest}best_recog_data.json")
        with open(
            f"./data/aishell/{task}/BertCompare/{setting}/{nbest}best_recog_data.json",
                "w"
        ) as f:
            json.dump(recog_dict, f, ensure_ascii=False, indent=4)
    
if (stage <= 2) and (stop_stage >= 2):
    print(f'rescoring')
    best_weight = 0.0
    with open(f"./data/aishell/dev/BertCompare/{setting}/{nbest}best_recog_data.json") as f:
        recog_file = json.load(f)

        # find best weight
        best_weight = 0.0
        min_err = 100
        for w in range(101):
            weight = w * 0.01
            for data in recog_file:
                first_score = torch.tensor(data['first_score'])
                rescore = torch.tensor(data['rescore'])
                cer = data['cer']
                cer = cer.view(-1, 4)

                weighted_score = first_score + weight * rescore

                max_index = torch.argmax(weighted_score).item()

                correction += cer[max_index][0]
                substitution += cer[max_index][1]
                deletion += cer[max_index][2]
                insertion += cer[max_index][3]

                err_for_weight = (substitution + deletion + insertion) / (
                        correction + deletion + substitution
                    )
                if (err_for_weight <= min_err):
                    print(f'better_weight:{min_weight}, smaller_err:{min_err}')
                    min_weight = weight
                    min_err = err_for_weight
                print(f'min_weight:{min_weight}, min_err:{min_err}')
        
    for task in recog_set:
        with open(f"./data/aishell/{task}/BertCompare/{setting}/{nbest}best_recog_data.json") as f:       
            recog_file = json.load(f)

            recog_dict = dict()
        recog_dict["utts"] = dict()
        for n, data in enumerate(recog_file):
            token = data["token"][:nbest]
            ref = data["ref"]

            score = torch.tensor(data["first_score"][:nbest])
            rescore = torch.tensor(data["rescore"][:nbest])

            weight_sum = score + best_weight * rescore

            max_index = torch.argmax(weight_sum).item()

            best_hyp = token[max_index]

            sep = best_hyp.index(102)
            best_hyp = tokenizer.convert_ids_to_tokens(t for t in best_hyp[1:sep])
            ref = list(ref[0])
            # remove [CLS] and [SEP]
            token_list = [str(t) for t in best_hyp]
            ref_list = [str(t) for t in ref]
            recog_dict["utts"][f"{task}_{n + 1}"] = dict()
            recog_dict["utts"][f"{task}_{n + 1}"]["output"] = {
                "rec_text": "".join(token_list),
                "rec_token": " ".join(token_list),
                "first_score": score.tolist(),
                "second_score": rescore.tolist(),
                "rescore": weight_sum.tolist(),
                "text": "".join(ref_list),
                "text_token": " ".join(ref_list),
            }

        with open(
            f"data/aishell/{task}/BertCompare/{setting}/{nbest}best_rescore_data.json",
            "w",
        ) as f:
            json.dump(recog_dict, f, ensure_ascii=False, indent=4)





            



import torch
import json
import sys
import logging
import wandb
from tqdm import tqdm
import numpy as np
sys.path.append("../")
from src_utils.LoadConfig import load_config
from model.combinePBERT import prepare_ensemble_pbert
from pathlib import Path
from torch.utils.data import DataLoader
from utils.Datasets import prepare_pBERT_Dataset, load_scoreData
from utils.CollateFunc import ensemblePBERTCollate
from RescoreBert.utils.CollateFunc import  NBestSampler, RescoreBert_BatchSampler
from RescoreBert.utils.PrepareScoring import prepareRescoreDict, prepare_score_dict, calculate_cer, get_result
from torch.optim.lr_scheduler import OneCycleLR

config = "/mnt/disk6/Alfred/Rescoring/src/Ensemble/config/ensemblePBERT.yaml"
args, train_args, recog_args = load_config(config)
setting = 'withLM' if args['withLM'] else 'noLM'

# Step 1: Load datasets
# Load every data.json at each task, making 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

methods = ['CLM', 'MLM','RescoreBert_MD', 'RescoreBert_MWER',  'RescoreBert_MWED', 'Bert_sem', 'BertAlsem'] # , 'MLM','RescoreBert_MD', 'RescoreBert_MWER',  'RescoreBert_MWED', 'Bert_sem', 'BertAlsem','Bert_alsem', , 
wer_weight = {
    'ASR': 7.34,
    'CLM': 6.05,
    'MLM': 5.17,
    'RescoreBert_MD': 5.29,
    'RescoreBert_MWER': 5.29,
    'RescoreBert_MWED': 5.30,
    'PBERT': 5.08,
    'PBert_LSTM': 5.04,
    'Bert_sem': 5.27,
    'BertAlsem': 5.25
}
train_score_dict = {}
valid_score_dict = {}
score_dicts = [train_score_dict, valid_score_dict]
datasets = {
    'train': None,
    'dev': None
}

dataset = ["train", "dev"]
weight_list = []
if (train_args['use_WER']):
    weight_list.append(10 - wer_weight['ASR'])
    for name in methods:
        weight_list.append( 10 -  wer_weight[name] )
    weights = torch.as_tensor(weight_list)
    from torch.nn import Softmax
    softmax = Softmax(dim = -1)
    weights = softmax(weights)
else:
    weight_list.append(1.0)
    for name in methods:
        weight_list.append(1.0)
    weights = torch.as_tensor(weight_list)

print(f'weights:{weights}')

model, tokenizer = prepare_ensemble_pbert(args, train_args, device = device ,feature_num = 3 + int(len(methods)))

for k, (task, score_dict) in enumerate(zip(dataset, score_dicts)):
    with open(f"/mnt/disk6/Alfred/Rescoring/data/{args['dataset']}/data/{setting}/{task}/data.json") as f:
        print(f'{task}:load org data')
        data_json = json.load(f)
        for data in data_json:
            if (not data['name'] in score_dict.keys()):
                score_dict[data['name']] = dict()
                score_dict[data['name']]['feature'] = [[] for _ in range(int(args['nbest']))]
                score_dict[data['name']]['feature_rank'] = [[] for _ in range(int(args['nbest']))]
                score_dict[data['name']]['hyps'] = data['hyps'][:int(args['nbest'])]
                score_dict[data['name']]['ref'] = data['ref']
                score_dict[data['name']]['wer'] = [wer['err'] for wer in data['err']]
            
            am_score = np.asarray(data['am_score'][:args['nbest']])
            if (train_args['use_Norm']):
                am_score = (am_score - am_score.min()) / (am_score.max() - am_score.min())
            am_score = am_score * weights[0].item()
            am_rank = torch.as_tensor(am_score).argsort(dim = -1, descending=True).tolist()
            for i, (score, rank) in enumerate(zip(am_score, am_rank)):
                score_dict[data['name']]['feature'][i].append(score)
                score_dict[data['name']]['feature_rank'][i].append(rank)

            ctc_score = np.asarray(data['ctc_score'][:args['nbest']])
            if (train_args['use_Norm']):
                ctc_score = (ctc_score - ctc_score.min()) / (ctc_score.max() - ctc_score.min())
            ctc_score = ctc_score * weights[0].item()

            ctc_rank = torch.as_tensor(ctc_score).argsort(dim = -1, descending=True).tolist()
            for i, (score, rank) in enumerate(zip(ctc_score, ctc_rank)):
                score_dict[data['name']]['feature'][i].append(score)
                score_dict[data['name']]['feature_rank'][i].append(rank)

            if (data['lm_score'] is not None):
                lm_score = np.asarray(data['lm_score'][:args['nbest']])
                if (train_args['use_Norm']):
                    lm_score = (lm_score - lm_score.min()) / (lm_score.max() - lm_score.min())
                lm_score = lm_score * weights[0].item()

                lm_rank = torch.as_tensor(lm_score).argsort(dim = -1, descending=True).tolist()
                for i, (score, rank) in enumerate(zip(lm_score, lm_rank)):
                    score_dict[data['name']]['feature'][i].append(score)
                    score_dict[data['name']]['feature_rank'][i].append(rank)
            else:
                for i, score in enumerate(data['am_score'][:args['nbest']]):
                    score_dict[data['name']]['feature'][i].append(0.0)
                    score_dict[data['name']]['feature_rank'][i].append(1 / (i + 1))


    print(f'{task}:load rescore data')
    data_path = Path(f"/mnt/disk6/Alfred/Rescoring/data/result/aishell/noLM/{task}/10best")
    for method, weight in zip(methods, weights[1:]):
        print(f'{task}: {method} loading')
        ensemble_path = f"{data_path}/{method}/data.json"

        with open(ensemble_path) as f:
            ensemble_data_json = json.load(f)
        
        print(f'datalen:{len(ensemble_data_json)}')
    
        score_dict = load_scoreData(
            ensemble_data_json,
            score_dict, 
            int(args['nbest']), 
            -1, 
            wer_weight=weight.item(), 
            use_Norm=train_args['use_Norm']
        )
    
    datasets[task] = prepare_pBERT_Dataset(score_dict, tokenizer, topk = 10)

print(f'type(train_dataset):{type(datasets["train"])}')
print(f'# of train:{len(train_score_dict.keys())},\n # of valid:{len(valid_score_dict.keys())}')

(
    index_dict,
    inverse_dict,
    am_scores,
    ctc_scores,
    lm_scores,
    rescores,
    wers,
    hyps,
    refs,
) = prepare_score_dict(data_json, nbest=args["nbest"])
rescores_flush = rescores.copy()

# check feature_num:
train_feature_num = -1
for name in tqdm(train_score_dict.keys()):
    # print(f"name:{name}, num:{len(train_score_dict[name]['feature'][0])}")
    if (train_feature_num > 0):
        if (train_args['use_rank']):
            assert(train_feature_num == len(train_score_dict[name]['feature_rank'][0])), f"{train_feature_num} != {len(train_score_dict[name]['feature_rank'][0])}"
        else:
            assert(train_feature_num == len(train_score_dict[name]['feature'][0])), f"{name} : {train_feature_num} != {len(train_score_dict[name]['feature'][0])}"

    if (train_args['use_rank']):
        train_feature_num = len(train_score_dict[name]['feature_rank'][0])
    else:
        train_feature_num = len(train_score_dict[name]['feature'][0])
    # take the score part in data_json

valid_feature_num = -1
for name in valid_score_dict.keys():
    if (valid_feature_num > 0):
        if (train_args['use_rank']):
            assert(valid_feature_num == len(valid_score_dict[name]['feature_rank'][0])), f"{valid_feature_num} != {len(valid_score_dict[name]['feature_rank'][0])}"
        else:
            assert(valid_feature_num == len(valid_score_dict[name]['feature'][0])), f"{valid_feature_num} != {len(valid_score_dict[name]['feature'][0])}"
    
    if (train_args['use_rank']):
        valid_feature_num = len(valid_score_dict[name]['feature_rank'][0])
    else:
        valid_feature_num = len(valid_score_dict[name]['feature'][0])

assert(train_feature_num == valid_feature_num), f"train:{train_feature_num}, valid:{valid_feature_num}"
# concat making a dataset

train_sampler = NBestSampler(datasets['train'])
valid_sampler = NBestSampler(datasets['dev'])

train_batch_sampler = RescoreBert_BatchSampler(
    train_sampler, train_args['train_batch']
)
valid_batch_sampler = RescoreBert_BatchSampler(valid_sampler, train_args['train_batch'])


train_loader = DataLoader(
    dataset=datasets['train'],
    batch_sampler=train_batch_sampler,
    collate_fn=ensemblePBERTCollate,
    # batch_size=train_args['train_batch'],
    num_workers=8,
    pin_memory=True
    )

valid_loader = DataLoader(
    dataset = datasets['dev'],
    batch_sampler = valid_batch_sampler,
    collate_fn=ensemblePBERTCollate,
    # batch_size=train_args['train_batch'],
    num_workers=8,
    pin_memory=True
    )


if (train_args['optim'] == 'adamW'):
    optimizer = torch.optim.AdamW(model.parameters(), lr = float(train_args['lr']))
elif (train_args['optim'] == 'adam'):
    optimizer = torch.optim.Adam(model.parameters(), lr = float(train_args['lr']))
elif (train_args['optim'] == 'sgd'):
    optimizer = torch.optim.SGD(model.parameters(), lr = float(train_args['lr']))


# lr_scheduler = OneCycleLR(
#     optimizer,
#     max_lr = float(train_args['lr']),
#     steps_per_epoch = len(train_loader),
#     epochs = int(train_args['epoch']),
#     pct_start = 0.01
# )

step = 0
optimizer.zero_grad(set_to_none=True)
wandb_config = wandb.config
wandb_config = {
    'args': args,
    'train_args': train_args
}


model_name = "PBERT"

run_name = f"{args['nbest']}Best_Ensembl{model_name}_batch{train_args['train_batch']}_accum{train_args['accumgrad']}grads_{train_args['optim']}_lr{train_args['lr']}_reduction{train_args['reduction']}_{train_feature_num}features"
optimizer.zero_grad(set_to_none=True)
path_name = f"./checkpoint/{args['dataset']}/Ensemble/{model_name}/{setting}/{args['nbest']}best/batch{train_args['train_batch']}_accum{train_args['accumgrad']}grads_{train_args['optim']}_lr{train_args['lr']}_reduction{train_args['reduction']}_{train_feature_num}features"
if (train_args['use_rank']):
    run_name += "_useRank"
    path_name += "_useRank"   
if (train_args['use_WER']):
    run_name += "_WER"
    path_name += "_WER"
if (train_args['use_Norm']):
    run_name += "_Norm"
    path_name += "_Norm"

wandb.init(
    project=f"Ensemble_{args['dataset']}_{setting}",
    config=wandb_config,
    name=run_name
)
checkpoint_path = Path(path_name)

checkpoint_path.mkdir(parents=True, exist_ok=True)
steps_per_epoch = len(train_batch_sampler) // int(train_args['accumgrad'])

wandb.watch(model, log_freq=int(train_args["print_loss"]))
model = model.to(device)
log_flag = True
min_cer = 1e6
logging_loss = 0.0
total_step = 0
for e in range(train_args['epoch']):
    train_epoch_loss = 0.0
    model.train()
    for i, data in enumerate(tqdm(train_loader)):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        feature = data['feature'].to(device)
        nBestIndex = data['nBestIndex'].to(device)
        labels = data['label'].to(device)

        # if (train_args['use_rank']):
        #     loss = model(feature_rank, nBestIndex, labels = labels)['loss']
        # else:
        loss = model(input_ids, attention_mask, nBestIndex, feature, labels = labels)['loss']
        
        # print(f'loss:{loss}')

        loss = loss / int(train_args['accumgrad'])
        loss.backward()

        # print(f'loss:{loss}')

        if ((i + 1) % int(train_args["accumgrad"])) == 0:
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            total_step += 1
            log_flag = True

        if step > 0 and (step % int(train_args["print_loss"]) == 0 and log_flag):
            logging.warning(f"step {step},loss:{logging_loss / step}")
            wandb.log(
                {
                    "train_loss": (logging_loss / step),
                },
                step= total_step,
            )
            logging_loss = torch.tensor([0.0], device=device)
            log_flag = False
        
        logging_loss += loss.clone().detach()
        train_epoch_loss += loss.clone().detach()
    checkpoint = {
        'epoch': e,
        'checkpoint': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args, 'train_args': train_args,
        "method": methods
    }
    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_{e + 1}.pt")

    print(f'eval')
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for j, data in enumerate(tqdm(valid_loader)):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            feature = data['feature'].to(device)
            nBestIndex = data['nBestIndex'].to(device)
            labels = data['label'].to(device)

            # if (train_args['use_rank']):
            #     output = model(feature_rank, nBestIndex, labels = labels)
            # else:
            output = model(input_ids, attention_mask, nBestIndex, feature, labels = labels)
            loss = output['loss']
            logit = output['logit']

            for name, index, score in zip(data['name'], data['index'],logit):
                rescores[index_dict[name]][index] = score.item()

            valid_loss += loss.item()

        best_am, best_ctc, best_lm, best_rescore, eval_cer = calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            am_range=[0, 1],
            ctc_range=[0, 1],
            lm_range=[0, 1],
            rescore_range=[0, 1],
            search_step=0.2,
            recog_mode=False,
        )

        print(f'epoch:{e+1} ,\n eval_loss:{valid_loss}  \neval_cer: {eval_cer}')
        wandb.log(
            {
                'valid_loss': (valid_loss / len(valid_batch_sampler)),
                'train_epoch_loss': (train_epoch_loss / steps_per_epoch),
                "epoch": e + 1,
                "eval_cer": eval_cer

            }, step = total_step
        )

    if eval_cer < min_cer:
        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_train_best_CER.pt")
        min_val_cer = eval_cer
    rescores = rescores_flush.copy()




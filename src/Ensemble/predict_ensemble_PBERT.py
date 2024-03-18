import torch
import json
import sys
import logging
import wandb
from tqdm import tqdm
sys.path.append("../")
from src_utils.LoadConfig import load_config
from model.combinePBERT import prepare_ensemble_pbert
from pathlib import Path
from torch.utils.data import DataLoader
from utils.Datasets import prepare_pBERT_Dataset, load_scoreData
from utils.CollateFunc import ensemblePBERTCollate
from RescoreBert.utils.CollateFunc import  NBestSampler, BatchSampler, RescoreBert_BatchSampler
from RescoreBert.utils.PrepareScoring import prepareRescoreDict, prepare_score_dict, calculate_cer, get_result

config = "/mnt/disk6/Alfred/Rescoring/src/Ensemble/config/ensemblePBERT.yaml"
args, train_args, recog_args = load_config(config)
setting = 'withLM' if args['withLM'] else 'noLM'

checkpoint_path = sys.argv[1]
checkpoint = torch.load(checkpoint_path)

methods = checkpoint['method']
print(f'method:{methods}')
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, tokenizer = prepare_ensemble_pbert(args, train_args, device = device ,feature_num = 3 + int(len(methods)))

recog_task = ['dev', 'test']
if (args['dataset'] in ['librispeech']):
    recog_task = ['dev', 'dev_clean', 'dev_other', 'test_clean', 'test_other']
if (args['dataset'] in ['aishell2']):
    recog_task = ['dev', 'test_ios', 'test_android', 'test_mic']
if (args['dataset'] in ['csj']):
    recog_task = ['dev', 'eval1', 'eval2', 'eval3']
for task in recog_task:
    score_dict = {}

    with open(f"/mnt/disk6/Alfred/Rescoring/data/{args['dataset']}/data/{setting}/{task}/data.json") as f:
        print(f'{task}:load org data')
        data_json = json.load(f)
        for data in data_json:
            if (not data['name'] in score_dict.keys()):
                score_dict[data['name']] = dict()
                score_dict[data['name']]['feature'] = [[] for _ in range(int(args['nbest']))]
                score_dict[data['name']]['hyps'] = data['hyps'][:int(args['nbest'])]
                score_dict[data['name']]['ref'] = data['ref']
                score_dict[data['name']]['wer'] = [wer['err'] for wer in data['err']]
            
            for i, score in enumerate(data['am_score'][:args['nbest']]):
                # print(f'i:{i}')
                # print(f"len:{len(score_dict[data['name']]['feature'])}")
                score_dict[data['name']]['feature'][i].append(score)
            for i, score in enumerate(data['ctc_score'][:args['nbest']]):
                score_dict[data['name']]['feature'][i].append(score)
            if (data['lm_score'] is not None):
                for i, score in enumerate(data['lm_score'][:args['nbest']]):
                    score_dict[data['name']]['feature'][i].append(score)
            else:
                for i, score in enumerate(data['am_score'][:args['nbest']]):
                    score_dict[data['name']]['feature'][i].append(0.0)
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

    print(f'{task}:load rescore data')
    data_path = Path(f"/mnt/disk6/Alfred/Rescoring/data/result/{args['dataset']}/{setting}/{task}/10best")
    for method in methods:
        print(f'{task}: {method} loading')
        ensemble_path = f"{data_path}/{method}/data.json"

        with open(ensemble_path) as f:
            ensemble_data_json = json.load(f)
    
        score_dict =  load_scoreData(
            ensemble_data_json,
            score_dict, 
            int(args['nbest']), 
            -1, 
            # wer_weight=weight.item(), 
            use_Norm=train_args['use_Norm']
        )
    
    feature_num = len(score_dict[data['name']]['feature'][0])
    
    recog_dataset =  prepare_pBERT_Dataset(score_dict, tokenizer, topk = 10)
    
    recog_sampler = NBestSampler(recog_dataset)
    recog_batch_sampler = RescoreBert_BatchSampler(recog_sampler, recog_args["batch"])
    recog_loader = DataLoader(
        dataset=recog_dataset,
        batch_sampler=recog_batch_sampler,
        collate_fn=ensemblePBERTCollate,
        num_workers=16,
    )

    model.load_state_dict(checkpoint['checkpoint'])
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(recog_loader)):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            feature = data['feature'].to(device)
            nBestIndex = data['nBestIndex'].to(device)

            output = model(input_ids, attention_mask, nBestIndex, feature, labels = None)['logit']

            for n, (name, index, score) in enumerate(
                    zip(data["name"], data["index"], output)
                ):
                    assert torch.isnan(score).count_nonzero() == 0, "got nan"
                    rescores[index_dict[name]][index] += score.item()
        
        if task == 'dev':
            best_am, best_ctc, best_lm, best_rescore, min_cer = calculate_cer(
                am_scores,
                ctc_scores,
                lm_scores,
                rescores,
                wers,
                am_range=[0, 1],
                ctc_range=[0, 1],
                lm_range=[0, 1],
                rescore_range=[0, 1],
                search_step=0.1,
            )

            print(
                f"am_weight:{best_am}\n ctc_weight:{best_ctc}\n lm_weight:{best_lm}\n rescore_weight:{best_rescore}\n CER:{min_cer}"
            )

        cer, result_dict = get_result(
                inverse_dict,
                am_scores,
                ctc_scores,
                lm_scores,
                rescores,
                wers,
                hyps,
                refs,
                am_weight=best_am,
                ctc_weight=best_ctc,
                lm_weight=best_lm,
                rescore_weight=best_rescore,
            )

        print(f"Dataset:{args['dataset']}")
        print(f"setting:{setting}")
        print(f"task:{task}")
        print(f"CER : {cer}")


    


    

    


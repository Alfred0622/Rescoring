import json
from jiwer import cer, wer
import random
import argparse
import numpy as np
from utils.PrepareScoring import prepare_score_dict, calculate_cer, get_result

parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--dataset', 
    help = "which dataset",
    type = str,
    default= 'aishell'
)

parser.add_argument(
    '-w', '--withLM', 
    help = "withLM or not",
    type = str,
    default= 'noLM'
)

parser.add_argument(
    '-n', '--name', 
    help = "save_name",
    type = str,
    default= 'noName'
)

parser.add_argument(
    '-s', '--seed', 
    help = "seed number",
    type = int,
    default= 42
)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
print(f"seed : {args.seed}")

recog_set = ['dev', 'test']
if (args.dataset in ['aishell2']):
    recog_set = ['dev_ios', 'test_ios', 'test_android', 'test_mic']
elif (args.dataset in ['csj']):
    recog_set = ['eval1', 'eval2', 'eval3']
elif (args.dataset in ['librispeech']):
    recog_set = ['valid', 'dev_clean', 'dev_other', 'test_clean', 'test_other']

best_am = 0.0
best_ctc = 0.0
best_lm = 0.0
best_rescore = 0.0

for recog_task in recog_set:
    with open(f"../../data/{args.dataset}/data/{args.withLM}/{recog_task}/data.json") as f:
        data_list = json.load(f)

        index_dict, inverse_dict,am_scores, ctc_scores, lm_scores, rescores, wers, hyps, refs = prepare_score_dict(data_list, nbest = 50)
        
        top_hyps = []
        # hyps = []
        refs = []
        for data in data_list:
            rand_score = np.random.uniform(low = -10, high = 0.1, size = len(data['hyps']))
            rescores[index_dict[data['name']]] = np.pad(rand_score,pad_width=(0, 50 - len(data['hyps'])) ,mode = 'constant', constant_values = -1e8)
            # print(f"am_scores:{am_scores[index_dict[data['name']]]}")
            # print(f"ctc_scores:{ctc_scores[index_dict[data['name']]]}")
            # print(f"rescores:{rescores[index_dict[data['name']]]}")
            # break
            # am_scores[index_dict[data['name']]][ am_scores[index_dict[data['name']]] == -np.inf] = -1e8
            # ctc_scores[index_dict[data['name']]][ ctc_scores[index_dict[data['name']]] == -np.inf] = -1e8
            # lm_scores[index_dict[data['name']]][ lm_scores[index_dict[data['name']]] == -np.inf] = -1e8
            # if (np.count_nonzero(am_scores[index_dict[data['name']]] == 0) > 0):
            #     # print(np.count_nonzero(np.isinf(am_scores[index_dict[data['name']]])))
            #     print('am:')
            #     print(am_scores[index_dict[data['name']]])
            # if (np.count_nonzero(ctc_scores[index_dict[data['name']]] == 0) > 0):
            #     print('ctc:')
            #     # print(np.count_nonzero(np.isinf(ctc_scores[index_dict[data['name']]])))
            #     print(ctc_scores[index_dict[data['name']]])
            # if (np.count_nonzero(lm_scores[index_dict[data['name']]] == 0) > 0):
            #     # print(np.count_nonzero(np.isinf(lm_scores[index_dict[data['name']]])))
            #     print('lm:')
            #     print(lm_scores[index_dict[data['name']]])

            top_hyps.append(data['hyps'][0])
            refs.append(data['ref'])
        
        print(np.count_nonzero(np.isinf(am_scores)))
        print(np.count_nonzero(np.isinf(ctc_scores)))
        print(np.count_nonzero(np.isinf(lm_scores)))
        print(np.count_nonzero(np.isinf(rescores)))

        if (recog_task in ['dev', 'dev_ios', 'valid']):
            best_am, best_ctc, best_lm, best_rescore, min_cer = calculate_cer(
                am_scores,
                ctc_scores,
                lm_scores,
                rescores,
                wers,
                am_range = [0, 1],
                ctc_range = [0, 1],
                lm_range = [0, 1],
                rescore_range = [0, 1],
                search_step = 0.1 
            )
        

        print(f'am_weight:{best_am}\n ctc_weight:{best_ctc}\n lm_weight:{best_lm}\n rescore_weight:{best_rescore}\n CER:{min_cer}')


        cer, result_dict = get_result(
            inverse_dict,
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            hyps,
            refs,
            am_weight = best_am,
            ctc_weight = best_ctc,
            lm_weight = best_lm,
            rescore_weight = best_rescore
        )

        print(f"{args.dataset} {args.withLM} {recog_task} : Origin WER = {wer(refs, top_hyps)}")
        print(f"{args.dataset} {args.withLM} {recog_task} : WER = {cer}\n\n")
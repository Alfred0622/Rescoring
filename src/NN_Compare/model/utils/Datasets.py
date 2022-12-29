from regex import P
from torch.utils.data import Dataset
from jiwer import wer, cer
import numpy as np
from tqdm import tqdm
class rescoreDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
    
    def __getitem__(self, index: int):
        return self.data_list[index]
    
    def __len__(self) -> int:
        return len(self.data_list)


def get_Dataset(data_json):
    data_list = list()

    for i, data in enumerate(tqdm(data_json)):
        wers = list()

        for hyp in data['hyps']:
            wers.append(wer([data['ref']], [hyp]))
        wers = np.array(wers)
        sort_wers = np.argsort(wers)
        # second lowest WER
        if (len(data['hyps_id'][sort_wers[0]]) == 0):
            continue
        if (len(data['hyps_id'][sort_wers[1]]) > 0):
            data_list.append(
                {
                    "input_1": [int(ids) for ids in data['hyps_id'][sort_wers[0]]],
                    "input_2": [int(ids) for ids in data['hyps_id'][sort_wers[1]]],
                    "labels": 1 if (wers[sort_wers[0]] <= wers[sort_wers[1]]) else 0,

                    "am_score": [data['am_score'][sort_wers[0]], data['am_score'][sort_wers[1]]],
                    "ctc_score": [data['ctc_score'][sort_wers[0]], data['ctc_score'][sort_wers[1]]],
                    "lm_score": [data['lm_score'][sort_wers[0]], data['lm_score'][sort_wers[1]]] if (data['lm_score'] != []) else [0, 0]
                }
            )

            data_list.append(
                {
                    "input_1": [int(ids) for ids in data['hyps_id'][sort_wers[1]]],
                    "input_2": [int(ids) for ids in data['hyps_id'][sort_wers[0]]],
                    "labels": 1 if (wers[sort_wers[1]] <= wers[sort_wers[0]]) else 0,
                    "am_score": [data['am_score'][sort_wers[1]], data['am_score'][sort_wers[0]]],
                    "ctc_score": [data['ctc_score'][sort_wers[1]], data['ctc_score'][sort_wers[0]]],
                    "lm_score": [data['lm_score'][sort_wers[1]], data['lm_score'][sort_wers[0]]] if (data['lm_score'] != []) else [0, 0]                
                }
            )
            
        # Highest WER
        if (len(data['hyps_id'][sort_wers[-1]]) > 0):
            data_list.append(
                {
                    "input_1": [int(ids) for ids in data['hyps_id'][sort_wers[0]]],
                    "input_2": [int(ids) for ids in data['hyps_id'][sort_wers[-1]]],
                    "labels": 1 if (wers[sort_wers[0]] <= wers[sort_wers[-1]]) else 0,

                    "am_score": [data['am_score'][sort_wers[0]], data['am_score'][sort_wers[-1]]],
                    "ctc_score": [data['ctc_score'][sort_wers[0]], data['ctc_score'][sort_wers[-1]]],
                    "lm_score": [data['lm_score'][sort_wers[0]], data['lm_score'][sort_wers[-1]]] if (data['lm_score'] != []) else [0, 0]                
                }
            )

            data_list.append(
                {
                    "input_1": [int(ids) for ids in data['hyps_id'][sort_wers[-1]]],
                    "input_2": [int(ids) for ids in data['hyps_id'][sort_wers[0]]],
                    "labels": 1 if (wers[sort_wers[-1]] <= wers[sort_wers[0]]) else 0,

                    "am_score": [data['am_score'][sort_wers[-1]], data['am_score'][sort_wers[0]]],
                    "ctc_score": [data['ctc_score'][sort_wers[-1]], data['ctc_score'][sort_wers[0]]],
                    "lm_score": [data['lm_score'][sort_wers[-1]], data['lm_score'][sort_wers[0]]] if (data['lm_score'] != []) else [0, 0]          
                }
            )
            
        # Top-1 ASR score
        if (len(data['hyps_id'][0]) > 0):
            data_list.append(
                {
                    "input_1": [int(ids) for ids in data['hyps_id'][sort_wers[0]]],
                    "input_2": [int(ids) for ids in data['hyps_id'][0]],
                    "labels": 1 if (wers[sort_wers[0]] <= wers[0]) else 0,

                    "am_score": [data['am_score'][sort_wers[0]], data['am_score'][0]],
                    "ctc_score": [data['ctc_score'][sort_wers[0]], data['ctc_score'][0]],
                    "lm_score": [data['lm_score'][sort_wers[0]], data['lm_score'][0]] if (data['lm_score'] != []) else [0, 0]     
                }
            )

            data_list.append(
                {
                    "input_1": [int(ids) for ids in data['hyps_id'][0]],
                    "input_2": [int(ids) for ids in data['hyps_id'][sort_wers[0]]],
                    "labels": 1 if (wers[0] <= wers[sort_wers[0]]) else 0,

                    "am_score": [data['am_score'][0], data['am_score'][sort_wers[0]]],
                    "ctc_score": [data['ctc_score'][0], data['ctc_score'][sort_wers[0]]],
                    "lm_score": [data['lm_score'][0], data['lm_score'][sort_wers[0]]] if (data['lm_score'] != []) else [0, 0]            
                }
            )

        # lowest ASR score
        if (len(data['hyps_id'][-1]) > 0):
            data_list.append(
                {
                    "input_1": [int(ids) for ids in data['hyps_id'][sort_wers[0]]],
                    "input_2": [int(ids) for ids in data['hyps_id'][-1]],
                    "labels": 1 if (wers[sort_wers[0]] <= wers[-1]) else 0,

                    "am_score": [data['am_score'][sort_wers[0]], data['am_score'][-1]],
                    "ctc_score": [data['ctc_score'][sort_wers[0]], data['ctc_score'][-1]],
                    "lm_score": [data['lm_score'][sort_wers[0]], data['lm_score'][-1]] if (data['lm_score'] != []) else [0, 0]     
                }
            )

            data_list.append(
                {
                    "input_1": [int(ids) for ids in data['hyps_id'][-1]],
                    "input_2": [int(ids) for ids in data['hyps_id'][sort_wers[0]]],
                    "labels": 1 if (wers[-1] <= wers[sort_wers[0]]) else 0,

                    "am_score": [data['am_score'][-1], data['am_score'][sort_wers[0]]],
                    "ctc_score": [data['ctc_score'][-1], data['ctc_score'][sort_wers[0]]],
                    "lm_score": [data['lm_score'][-1], data['lm_score'][sort_wers[0]]] if (data['lm_score'] != []) else [0, 0]            
                }
            )

    return rescoreDataset(data_list)
        

        
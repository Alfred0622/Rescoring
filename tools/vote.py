import json
import numpy as np
import os
from tqdm import tqdm
from nBestAligner.nBestAlign import align, alignNbest

if __name__ == "__main__":
    data = None
    recog_data = ['dev', 'test']
    for task in recog_data:
        print(f'{task}')
        with open(f"./data/aishell_{task}/dataset.json") as f:
            data = json.load(f)

        # align data
        vote_dict = dict()
        vote_dict['utts'] = dict()
        for i, d in enumerate(tqdm(data)):
            """
            For token:
                "[CLS]你好謝謝小籠包再見[SEP]"  
                -> ['你', '好', '謝', '謝', '小', '籠', '包', '再', '見']
            """
            token = [ [char for char in string[5:-5]] for string in d['token'] ]
            ref = [char for char in d['ref'][5:-5]]

            nbest = align(token)

            alignResult = alignNbest(nbest)

            if (task == 'test' and i == 384):
                print('nbest')
                for n in nbest:
                    print(n)
                
                print(f'alignresult')
                for n in alignResult:
                    print(n)


            vote_result = []
            # vote data
            nbest = [[] for _ in range(len(alignResult[0]))]
            for result in alignResult:
                for j, k in enumerate(result):
                    nbest[j].append(k)
                max_token = max(result, key = result.count)
                if (max_token != '-'):
                    vote_result.append(max_token)
            nbest_str = [" ".join(token) for token in nbest]
            vote_dict['utts'][f'{task}_{i + 1}'] = dict()
            vote_dict['utts'][f'{task}_{i + 1}']['output'] = {
                'nbest_align': nbest_str,
                'rec_token' : " ".join(vote_result),
                'text_token': " ".join(ref)
            }
            

        if (not os.path.exists(f"./data/aishell_{task}/vote")):
            os.mkdir(f"./data/aishell_{task}/vote")
        with open(f"./data/aishell_{task}/vote/vote.json", 'w') as f:
            json.dump(vote_dict, f, ensure_ascii=False, indent = 4)

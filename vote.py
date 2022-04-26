from cmath import cos
import json
import numpy as np
import os
from tqdm import tqdm

def align(nbest):
    """
    input : 
    [
        [a,b,c,d]  # hyp1
        [a,b,d,e]  # hyp2
        [a,b,c,d,e]# hyp3
    ]

    output:
    [
        [
            [a,a], [b,b], [c, -], [d,d], [-, e]
        ],
        [
            [a,a], [b,b]. [c,c], [d,d], [-, e]
        ]
    ]
    """
    pair = []
    for candidate in nbest[1:]:
        pair.append([nbest[0], candidate])
    
    # First, align them pair by pair -- using minimum edit distance
    first_align = []
    for a in pair:
        cost_matrix  = [[0 for i in range(len(a[1]) + 1)] for j in range(len(a[0]) + 1)]
        for j in range(len(a[1])):
            cost_matrix[0][j] = j
        for i in range(len(a[0])):
            cost_matrix[i][0] = i

        for i in range(0, len(a[0])):
            for j in range(0, len(a[1])):
                if (a[0][i] == a[1][j]):
                    cost_matrix[i + 1][j + 1] = min(
                        cost_matrix[i][j],
                        cost_matrix[i + 1][j] + 1,
                        cost_matrix[i][j + 1] + 1,
                    )
                else:
                    cost_matrix[i + 1][j + 1] = min(
                        cost_matrix[i][j] + 2,
                        cost_matrix[i + 1][j] + 1,
                        cost_matrix[i][j + 1] + 1,
                    )
            
        l1 = len(a[0]) - 1
        l2 = len(a[1]) - 1
        align_result = []
        while(l1 >= 0 and l2 >= 0):
            if (a[0][l1] == a[1][l2]):
                cost = 0
            else:
                cost = 2
            
            r = cost_matrix[l1 + 1][l2 + 1]
            diag = cost_matrix[l1][l2]
            left = cost_matrix[l1 + 1][l2]
            down = cost_matrix[l1][l2 + 1]

            if (r == diag + cost):
                align_result = [[a[0][l1], a[1][l2]]] + align_result
                l1 -= 1
                l2 -= 1
            
            else:
                if (r == left + 1):
                    align_result = [['-', a[1][l2]]] + align_result
                    l2 -= 1
                else:
                    align_result = [[a[0][l1], '-']] + align_result
                    l1 -= 1
        first_align.append(align_result)

    return first_align

                                
def alignNbest(nbestAlign):

    alignResult = nbestAlign[0]
    for a in nbestAlign[1:]:
        ali = [alignResult, a]
        l1 = 0
        l2 = 0
        align = []
        # print(f'ali:')
        # for l in ali:
        #     print(l)
        while(l1 < len(ali[0]) and l2 < len(ali[1])):
            if (ali[0][l1][0] == ali[1][l2][0]):
                align.append(ali[0][l1] + ali[1][l2][1:])
                l1 += 1
                l2 += 1
            else:
                if (ali[0][l1][0] == '-'):
                    align.append(['-'] + ali[0][l1][1:] + ['-' for _ in range(len(ali[1][l2]) - 1)])
                    l1 += 1
                else:
                    align.append(['-'] + ['-' for _ in range(len(ali[0][l1]) - 1)] + ali[1][l2][1:])
                    l2 += 1
            
            if (l1 == len(ali[0])):
                while(l2 < len(ali[1])):
                    align.append(['-'] + ['-' for _ in range (len(ali[0][l1 - 1]) - 1)] + ali[1][l2][1:])
                    l2 += 1
            elif (l2 == len(ali[1])):
                while(l1 < len(ali[0])):
                    align.append(['-'] + ali[0][l1][1:] + ['-' for _ in range (len(ali[1][l2 - 1]) - 1)])
                    l1 += 1
            
        alignResult = align
    
    return alignResult


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

from turtle import up
from tqdm import tqdm
import torch


import numpy as np


def find_weight_simp(data, bound = [0, 1]):
    min_cer = 1e8
    best_weight = 0.0
    lower_bound = bound[0]
    upper_bound = bound[1]
    print(f'(lower, upper) = ({lower_bound}, {upper_bound})')

    for weight in tqdm(np.arange(lower_bound, upper_bound, step = 0.01)):
        c = 0
        s = 0
        d = 0
        i = 0

        for key in data.keys():
            if (not isinstance(data[key]['score'], torch.Tensor)):
                data[key]['score'] = torch.tensor(data[key]['score'], dtype = torch.float64)
                data[key]['Rescore'] = torch.tensor(data[key]['Rescore'], dtype = torch.float64)
            score = (1 - weight) * data[key]['score'] + weight * data[key]['Rescore']

            best_index = torch.argmax(score)
            
            c += data[key]['err'][best_index]['hit']
            s += data[key]['err'][best_index]['sub']
            d += data[key]['err'][best_index]['del']
            i += data[key]['err'][best_index]['ins']

        cer = (s + d + i) / (c + s + d)
        
        # print(f'weight:{weight}, score:{score}')

        print(f'weight:{weight}, CER:{cer}')

        if (cer < min_cer):
            best_weight = weight
            min_cer = cer

    print(f'best weight:{best_weight},Min CER:{min_cer}')
    return best_weight

def find_weight(data, bound = [0, 1], ctc_weight = 0.5):
    # data : Dict
    # data must include: err and score
    min_cer = 1e8
    best_ctc_weight = 0.0
    best_lm_weight = 0.0
    best_weight = 0.0

    lower_bound = bound[0]
    upper_bound = bound[1]

    am_weight = 1 - ctc_weight

    for lm_weight in np.arange(lower_bound, upper_bound, step = 0.01):
        for weight in np.arange(lower_bound, upper_bound, step = 0.01):

            print(f'\ram_weight:{am_weight}, ctc_weight:{ctc_weight}, lm_weight:{lm_weight}, Rescore Weight:{weight}', end = "", flush = True)
            c = 0
            s = 0
            d = 0
            i = 0

            for key in data.keys():

                if ('am_score' in data[key].keys() and not isinstance(data[key]['am_score'], torch.Tensor)):
                    data[key]['am_score'] = torch.tensor(data[key]['am_score'], dtype = torch.float64)

                if ('ctc_score' in data[key].keys() and not isinstance(data[key]['ctc_score'], torch.Tensor)):
                    data[key]['ctc_score'] = torch.tensor(data[key]['ctc_score'], dtype = torch.float64)

                data[key]['Rescore'] = torch.tensor(data[key]['Rescore'], dtype = torch.float64)

                if ('lm_score' in data[key].keys() and len(data[key]['lm_score'] )> 0):
                    if ('lm_score' in data[key].keys() and not isinstance(data[key]['lm_score'], torch.Tensor)):
                        data[key]['lm_score'] = torch.tensor(data[key]['lm_score'], dtype = torch.float64)

                    score = (
                        (1 - ctc_weight) * data[key]['am_score'] + ctc_weight * data[key]['ctc_score']
                    ) + lm_weight * data[key]['lm_score'] + weight * data[key]['Rescore']
                else:
                    score = (
                        am_weight * data[key]['am_score'] + ctc_weight * data[key]['ctc_score']
                    )+ weight * data[key]['Rescore']

                best_index = torch.argmax(score)

                c += data[key]['err'][best_index]['hit']
                s += data[key]['err'][best_index]['sub']
                d += data[key]['err'][best_index]['del']
                i += data[key]['err'][best_index]['ins']

            cer = (s + d + i) / (c + s + d)

            if (cer < min_cer):
                min_cer = cer
                best_ctc_weight = ctc_weight
                best_lm_weight = lm_weight
                best_weight = weight
    
    print(f'final weight:{best_ctc_weight}, {best_lm_weight}, {best_weight}')
    
    return best_lm_weight, best_weight

# def find_weight_complex(data, alpha_bound = [0, 10], beta_bound = [8, 10], gamma_bound = [80, 100]):
#     # In find_weight_complex, data(Dict) neeeds to include : 'am_score', 'ctc_score' and 'lm_score'(Optional)
    
#     min_cer = 1e8
#     best_alpha = 1
#     best_beta = 1
#     best_gamma = 1

#     for alpha in torch.arange(alpha_bound[0], alpha_bound[1], step = 0.01):
#         for beta in torch.arange(beta_bound[0], beta_bound[1], step = 0.01):
#             for gamma in torch.arange(gamma_bound[0], gamma_bound[1], step = 0.01):
#                 c = 0
#                 s = 0
#                 d = 0
#                 i = 0
#                 for key in data.keys():
#                     score = alpha * data[key]['am_score'] + \
#                             (1 - alpha) * data[key]['ctc_score'] + \
#                             beta * data[key]['lm_score'] + gamma * data[key]['Rescore']
                    
#                     best_index = torch.argmax(score)

#                     c += data[key]['err'][best_index][0]
#                     s += data[key]['err'][best_index][1]
#                     d += data[key]['err'][best_index][2]
#                     i += data[key]['err'][best_index][3]
#                 cer = (s + d + i) / (c + s + d)

#                 if (cer < min_cer):
#                     best_alpha = alpha
#                     best_beta = beta
#                     best_gamma = gamma
#                     min_cer = cer
    
#     return best_alpha, best_beta, best_gamma


    



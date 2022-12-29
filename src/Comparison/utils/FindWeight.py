from turtle import up
import tqdm
import torch

import numpy as np

def find_weight(data, bound = [0, 10]):
    # data : Dict
    # data must include: err and score
    min_cer = 1e8
    best_weight = 1.0

    lower_bound = bound[0]
    upper_bound = bound[1]

    for weight in tqdm(np.arange(lower_bound, upper_bound, step = 0.01)):
        c = 0
        s = 0
        d = 0
        i = 0
        for key in data.keys():
            score = data[key]['score'] + weight * data[key]['Rescore']

            best_index = torch.argmax(score)

            c += data[key]['err'][best_index]['hit']
            s += data[key]['err'][best_index]['sub']
            d += data[key]['err'][best_index]['del']
            i += data[key]['err'][best_index]['ins']
        
        cer = (s + d + i) / (c + s + d)

        if (cer < min_cer):
            min_cer = cer
            best_weight = weight
    
    print(f'Best Weight:{best_weight}')
    return best_weight

def find_weight_complex(data, alpha_bound = [0, 10], beta_bound = [8, 10], gamma_bound = [80, 100]):
    # In find_weight_complex, data(Dict) neeeds to include : 'am_score', 'ctc_score' and 'lm_score'(Optional)
    
    min_cer = 1e8
    best_alpha = 1
    best_beta = 1
    best_gamma = 1

    for alpha in torch.arange(alpha_bound[0], alpha_bound[1], step = 0.01):
        for beta in torch.arange(beta_bound[0], beta_bound[1], step = 0.01):
            for gamma in torch.arange(gamma_bound[0], gamma_bound[1], step = 0.01):
                c = 0
                s = 0
                d = 0
                i = 0
                for key in data.keys():
                    score = alpha * data[key]['am_score'] + \
                            (1 - alpha) * data[key]['ctc_score'] + \
                            beta * data[key]['lm_score'] + gamma * data[key]['Rescore']
                    
                    best_index = torch.argmax(score)

                    c += data[key]['err'][best_index][0]
                    s += data[key]['err'][best_index][1]
                    d += data[key]['err'][best_index][2]
                    i += data[key]['err'][best_index][3]
                cer = (s + d + i) / (c + s + d)

                if (cer < min_cer):
                    best_alpha = alpha
                    best_beta = beta
                    best_gamma = gamma
                    min_cer = cer
    
    return best_alpha, best_beta, best_gamma


    



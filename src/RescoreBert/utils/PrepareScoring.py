import numpy as np
from numba import jit, njit
from numba.typed import List

def prepare_score_dict_simp(data_json, nbest):
    """
    for simp, we only use score only
    """
    index_dict = dict()
    scores = []
    rescores = []
    wers = []
    if (isinstance(data_json, list)):
        
        for i, data in enumerate(data_json):
            index_dict[data['name']] = i
            scores.append(np.array(data['score'][:nbest], dtype = np.float32))

            rescores.append(np.zeros(scores[-1].shape, dtype = np.float32))

            utt_wer = [[value for value in wer.values()] for wer in data['err']] # [single_wer, c, s, d, i]
            wers.append(np.array(utt_wer))
 
    elif (isinstance(data_json, dict)):
        
        for i, key in enumerate(data_json.keys()):
            index_dict[key] = i
            scores.append(np.array(data_json[key]['score'][:nbest], dtype = np.float32))

            rescores.append(np.zeros(scores[-1].shape, dtype = np.float32))

            utt_wer = [[value for value in wer.values()] for wer in data[key]['err']]

            wers.append(np.array(utt_wer))
    
    return index_dict, scores, rescores, wers


def prepare_score_dict(data_json, nbest):
    """
    we use am_score, lm_score, ctc_score here
    """
    index_dict = dict()

    am_scores = []
    lm_scores = []
    ctc_scores = []
    rescores = []
    wers = []

    if (isinstance(data_json, list)):
        """
        if data_json is a list, the format should be
        [
            {
                name: str,
                hyps: [],
                am_score:[],
                ...
            },
        ]
        """

        for i, data in enumerate(data_json):
            index_dict[data['name']] = i

            am_scores.append(
                np.array(data['am_score'][:nbest])
            )
            
            lm_scores.append(
                np.array(data['lm_score'][:nbest])
                if ('lm_score' in data.keys() and data['lm_score'] != None and len(data['lm_score']) > 0) else np.zeros(
                    am_scores[-1].shape[0], dtype = np.float32
                )
            )

            ctc_scores.append(
                np.array(data['am_score'][:nbest])
                if 'ctc_score' in data.keys() else np.zeros(
                    am_scores[-1].shape[0], dtype = np.float32
                )
            )

            rescores.append(
                np.zeros(am_scores[-1].shape[0], dtype = np.float32)
            )

            utt_wer = [[value for value in wer.values()] for wer in data['err']]
            wers.append(np.array(utt_wer))

        return index_dict, List(am_scores), List(ctc_scores), List(lm_scores), List(rescores), List(wers)
        

    elif (isinstance(data_json, dict)):
        """
        if data_json is a dict, the format should be
        name:{
            hyps:[],
            am_score:[],
            ...
        }
        """
         
        for i, key in enumerate(data_json.keys()):
            index_dict[key] = i

            am_scores.append(
                np.array(data_json[key]['am_score'][:nbest])
            )

            lm_scores.append(
                np.array(data_json[key]['am_score'][:nbest])
                if 'lm_score' in data_json[key].keys() else np.zeros(
                    am_scores[-1].shape[0], dtype = np.float32
                )
            )

            ctc_scores.append(
                np.array(data_json[key]['am_score'][:nbest])
                if 'ctc_score' in data_json[key].keys() else np.zeros(
                    am_scores[-1].shape[0], dtype = np.float32
                )
            )

            rescores.append(
                np.zeros(am_scores[-1].shape[0], dtype = np.float32)
            )

            utt_wer = [[value for value in wer.values()] for wer in data_json[key]['err']]
            wers.append(np.array(utt_wer))
        
        else:
            return index_dict, List(am_scores), List(ctc_scores), List(lm_scores), List(rescores), List(wers)

@jit()
def get_result_simp(scores, rescores, wers, alpha, beta):
    c = np.int64(0.0)
    s = np.int64(0.0)
    d = np.int64(0.0)
    i = np.int64(0.0)

    alpha = np.float64(alpha)
    beta = np.float64(beta)
            
    for score, rescore, wer in zip(scores, rescores, wers):
        total_score = (
            alpha * score + beta * rescore
        )

        max_index = np.argmax(total_score)

        c += wer[max_index][1]
        s += wer[max_index][2]
        d += wer[max_index][3]
        i += wer[max_index][4]

    cer = (s + d + i) / (c + s + d)

    return c,s,d,i, cer

# @jit()
def calculate_cer_simp(scores, rescores, wers, alpha_range = [0, 10], beta_range = [0,10], search_step = 0.1, cer = 100):
    min_cer = np.float64(cer)

    print(f'search_step = {search_step}')

    # assert isinstance(alpha_range, list) and len(alpha_range) == 2 , \
    #      "The type of alpha_range must be list and its length must be 2"
    # assert isinstance(beta_range, list) and len(beta_range) == 2 , \
    #      "The type of beta_range must be list and its length must be 2"

    best_alpha = alpha_range[0]
    best_beta = beta_range[0]

    alpha_lower, alpha_upper = alpha_range
    beta_lower, beta_upper = beta_range
    
    alpha_range = np.array(alpha_range)
    beta_range = np.array(beta_range)

    print(f'alpha:{alpha_range}')
    print(f'beta: {beta_range}')

    for alpha in np.arange(alpha_range[0], alpha_range[1] + 0.01 , step = search_step ):
        for beta in np.arange(beta_range[0], beta_range[1] + 0.01, step = search_step):
            c = np.int64(0.0)
            s = np.int64(0.0)
            d = np.int64(0.0)
            i = np.int64(0.0)
            for score, rescore, wer in zip(scores, rescores, wers):
                total_score = (
                    alpha * score + beta * rescore
                )

                max_index = np.argmax(total_score)

                c += wer[max_index][1]
                s += wer[max_index][2]
                d += wer[max_index][3]
                i += wer[max_index][4]
            cer = (s + d + i) / (c + s + d)

            if (min_cer > cer):
                best_alpha = alpha.copy()
                best_beta = beta.copy()

                print(f'alpha:{alpha}, beta:{beta}, cer:{cer}')

                   
                alpha_lower = (alpha - search_step) if alpha - search_step >= 0 else 0
                alpha_upper = alpha + search_step

                beta_lower = (beta - search_step) if (beta - search_step >= 0) else 0
                beta_upper = beta + search_step

                min_cer = cer
    
    print(f'best_alpha:{best_alpha}, best_beta:{best_beta}, min_cer:{min_cer}')
    
    if (search_step <= 0.01):
        return best_alpha, best_beta, min_cer
    elif (search_step <= 0.05):
        return calculate_cer_simp(
            scores,
            rescores,
            wers,
            alpha_range = [alpha_lower, alpha_upper],
            beta_range = [beta_lower, beta_upper],
            search_step = 0.01,
            cer = min_cer
        )
    elif (search_step <= 0.1):
        return calculate_cer_simp(
            scores,
            rescores,
            wers,
            alpha_range = [alpha_lower, alpha_upper],
            beta_range = [beta_lower, beta_upper],
            search_step = 0.05,
            cer = min_cer
        )
    else:
        return calculate_cer_simp(
            scores,
            rescores,
            wers,
            alpha_range = [alpha_lower, alpha_upper],
            beta_range = [beta_lower, beta_upper],
            search_step = search_step * 0.5,
            cer = min_cer
        )


def get_result(
    am_scores,
    ctc_scores,
    lm_scores,
    rescores,
    wers,
    am_weight,
    ctc_weight,
    lm_weight,
    rescore_weight
):
    c = np.int64(0.0)
    s = np.int64(0.0)
    d = np.int64(0.0)
    i = np.int64(0.0)

    am_weight = np.around(np.float64(am_weight), 2)
    lm_weight = np.around(np.float64(lm_weight), 2)
    ctc_weight = np.around(np.float64(ctc_weight), 2)
    rescore_weight = np.around(np.float(rescore_weight), 2)


    print(f'get result weight:\n am = {am_weight},\n ctc = {ctc_weight}, \n lm = {lm_weight}, \n rescore = {rescore_weight}')

    for am_score, ctc_score, lm_score, rescore, wer in zip(am_scores, ctc_scores, lm_scores ,rescores, wers):
        
        total_score = (
            am_weight * am_score + \
            ctc_weight * ctc_score + \
            lm_weight * lm_score + \
            rescore_weight * rescore
        )

        max_index = np.argmax(total_score)

        c += wer[max_index][1]
        s += wer[max_index][2]
        d += wer[max_index][3]
        i += wer[max_index][4]
    
    print(f'Result c:{c}, Result s:{s}, Result d:{d}, Result i:{i}')
    cer = (s + d + i) / (c + s + d)

    return cer


# @jit()
def calculate_cer(
    am_scores,
    ctc_scores,
    lm_scores,
    rescores, 
    wers, 
    am_range = np.array([0, 10]), 
    ctc_range = np.array([0, 10]), 
    lm_range = np.array([0, 10]), 
    rescore_range = np.array([0, 10]),
    search_step = 0.1,
    min_cer = 100,
    best_am = 0.0,
    best_ctc = 0.0,
    best_lm = 0.0,
    best_rescore = 0.0,
    first_flag = True
):
    
    if (first_flag):
        am_lower, am_upper = am_range
        ctc_lower, ctc_upper = ctc_range
        lm_lower, lm_upper = lm_range
        rescore_lower, rescore_upper = rescore_range
    
    else:
        am_lower = best_am - search_step
        am_upper = best_am + search_step
        ctc_lower = best_ctc - search_step 
        ctc_upper = best_ctc + search_step
        lm_lower = best_lm - search_step
        lm_upper = best_lm + search_step
        rescore_lower = best_rescore - search_step
        rescore_upper = best_rescore + search_step
    
    min_cer = np.float64(min_cer)

    print(f'\nsearch_step:{search_step}')
    print(f'am_range:{am_lower}, {am_upper}')
    print(f'ctc_range:{ctc_lower}, {ctc_upper}')
    print(f'lm_range:{lm_lower}, {lm_upper}')
    print(f'rescore_range:{rescore_lower}, {rescore_upper}')
    print(f'first_flag:{first_flag}')
    print(f'min cer:{min_cer}')

    for am_weight in np.arange(am_lower, am_upper + search_step, step = search_step):
        for ctc_weight in np.arange(ctc_lower, ctc_upper + search_step, step = search_step):
            for lm_weight in np.arange(lm_lower, lm_upper + search_step, step = search_step):
                for rescore_weight in np.arange(rescore_lower, rescore_upper + search_step, step = search_step):
                    
                    c = np.int64(0.0)
                    s = np.int64(0.0)
                    d = np.int64(0.0)
                    i = np.int64(0.0)

                    am_weight = np.around(am_weight, 2)
                    ctc_weight = np.around(ctc_weight, 2)
                    lm_weight = np.around(lm_weight, 2)
                    rescore_weight = np.around(rescore_weight, 2)

                    for am_score, ctc_score, lm_score, rescore, wer in zip(
                        am_scores, ctc_scores, lm_scores, rescores, wers
                    ):
                        total_score = (
                            am_weight * am_score + \
                            ctc_weight * ctc_score + \
                            lm_weight * lm_score + \
                            rescore_weight * rescore
                        )

                        max_index = np.argmax(total_score)

                        c += wer[max_index][1]
                        s += wer[max_index][2]
                        d += wer[max_index][3]
                        i += wer[max_index][4]
                    
                    cer = (s + d + i) / (c + s + d)
                    # print(f'am_weight:{am_weight}, ctc_weight:{ctc_weight}, lm_weight:{lm_weight}, rescore_weight:{rescore_weight}, cer:{cer}')
                    
                    if (min_cer > cer):
                        best_am = am_weight
                        best_ctc = ctc_weight
                        best_lm = lm_weight
                        best_rescore = rescore_weight
                        print(f'Old cer:{min_cer}')
                        min_cer = cer
 
                        print(f'New am:{best_am}, New ctc:{best_ctc}, New lm:{best_lm}, New rescore:{best_rescore}')
                        print(f'New cer:{cer}\n')
    
    if (search_step <= 0.01): 
        return best_am, best_ctc, best_lm, best_rescore, min_cer

    elif (search_step <= 0.05):
        return calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            search_step = 0.01,
            min_cer = min_cer,
            best_am = best_am,
            best_ctc = best_ctc,
            best_lm = best_lm,
            best_rescore = best_rescore,
            first_flag=False
        )
    
    elif (search_step <= 0.1):
        return calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            search_step = 0.05,
            min_cer = min_cer,
            best_am = best_am,
            best_ctc = best_ctc,
            best_lm = best_lm,
            best_rescore = best_rescore,
            first_flag=False
        )
    else: # search_step = 0.2
        return calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            search_step = search_step * 0.5,
            min_cer = min_cer,
            best_am = best_am,
            best_ctc = best_ctc,
            best_lm = best_lm,
            best_rescore = best_rescore,
            first_flag=False
        )
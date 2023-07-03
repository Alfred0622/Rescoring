import numpy as np
from numba import jit, njit
from numba.typed import List
import torch
from tqdm import tqdm
from jiwer import wer, cer
from itertools import zip_longest


def prepare_score_dict_simp(data_json, nbest):
    """
    for simp, we only use score only
    """
    index_dict = dict()
    scores = []
    rescores = []
    wers = []
    if isinstance(data_json, list):
        print(f"list")

        for i, data in enumerate(data_json):
            index_dict[data["name"]] = i
            scores.append(np.array(data["score"][:nbest], dtype=np.float32))

            rescores.append(np.zeros(scores[-1].shape, dtype=np.float32))

            utt_wer = [
                [value for value in wer.values()] for wer in data["err"]
            ]  # [single_wer, c, s, d, i]
            wers.append(np.array(utt_wer))

    elif isinstance(data_json, dict):
        print(f"dict")

        for i, key in enumerate(data_json.keys()):
            index_dict[key] = i
            scores.append(np.array(data_json[key]["score"][:nbest], dtype=np.float32))

            rescores.append(np.zeros(scores[-1].shape, dtype=np.float32))

            utt_wer = [
                [value for value in wer.values()] for wer in data_json[key]["err"]
            ]

            wers.append(np.array(utt_wer))

    return index_dict, scores, rescores, wers


def prepare_score_dict(data_json, nbest):
    """
    we use am_score, lm_score, ctc_score here
    """
    index_dict = dict()
    inverse_index = dict()

    am_scores = []
    lm_scores = []
    ctc_scores = []
    rescores = []
    wer_rescores = []
    wers = []
    hyps = []
    refs = []

    if isinstance(data_json, list):
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
        print(f"list")

        for i, data in enumerate(data_json):
            index_dict[data["name"]] = i
            inverse_index[i] = data["name"]

            am_scores.append(data["am_score"][:nbest])

            legal_length = (
                len(am_scores[-1])
                if isinstance(am_scores[-1], list)
                else am_scores[-1].shape[0]
            )

            lm_scores.append(
                data["lm_score"][:nbest]
                if (
                    "lm_score" in data.keys()
                    and data["lm_score"] != None
                    and len(data["lm_score"]) > 0
                )
                else [0.0 for _ in range(legal_length)]
                # np.zeros(
                #     am_scores[-1].shape[0], dtype = np.float32
                # )
            )

            ctc_scores.append(
                data["ctc_score"][:nbest]
                if "ctc_score" in data.keys()
                else [0.0 for _ in range(legal_length)]
                # np.zeros(
                #     am_scores[-1].shape[0], dtype = np.float32
                # )
            )

            rescores.append([0.0 for _ in range(legal_length)])

            wer_rescores.append([0.0 for _ in range(legal_length)])

            utt_wer = [[value for value in wer.values()] for wer in data["err"]]
            wers.append(np.array(utt_wer))

            hyps.append(data["hyps"][:nbest])

            refs.append(data["ref"])

    elif isinstance(data_json, dict):
        """
        if data_json is a dict, the format should be
        name:{
            hyps:[],
            am_score:[],
            ...
        }
        """

        print(f"dict")

        for i, key in enumerate(data_json.keys()):
            index_dict[key] = i
            inverse_index[i] = key

            am_scores.append(data_json[key]["am_score"][:nbest])

            legal_length = (
                len(am_scores[-1])
                if isinstance(am_scores[-1], list)
                else am_scores[-1].shape[0]
            )

            lm_scores.append(
                data_json[key]["lm_score"][:nbest]
                if (
                    "lm_score" in data_json[key].keys()
                    and data_json[key]["lm_score"] != None
                    and len(data_json[key]["lm_score"]) > 0
                )
                else np.zeros(legal_length, dtype=np.float32)
            )

            ctc_scores.append(
                data_json[key]["ctc_score"][:nbest]
                if (
                    "lm_score" in data_json[key].keys()
                    and data_json[key]["ctc_score"] != None
                    and len(data_json[key]["ctc_score"]) > 0
                )
                else np.zeros(legal_length, dtype=np.float32)
            )

            rescores.append(
                [0.0 for _ in range(legal_length)]
                # np.zeros(am_scores[-1].shape[0], dtype = np.float32)
            )

            wer_rescores.append(
                [0.0 for _ in range(legal_length)]
                # np.zeros(am_scores[-1].shape[0], dtype = np.float32)
            )

            utt_wer = [
                [value for value in wer.values()] for wer in data_json[key]["err"]
            ]

            wers.append(np.array(utt_wer))

            hyps.append(data_json[key]["hyps"][:nbest])

            refs.append(data_json[key]["ref"])

    am_scores = np.array(
        list(zip_longest(*am_scores, fillvalue=np.NINF)), dtype=np.float32
    )
    ctc_scores = np.array(
        list(zip_longest(*ctc_scores, fillvalue=np.NINF)), dtype=np.float32
    )
    lm_scores = np.array(
        list(zip_longest(*lm_scores, fillvalue=np.NINF)), dtype=np.float32
    )
    rescores = np.array(
        list(zip_longest(*rescores, fillvalue=np.NINF)), dtype=np.float32
    )
    wer_rescores = np.array(
        list(zip_longest(*wer_rescores, fillvalue=np.NINF)), dtype=np.float32
    )

    return (
        index_dict,
        inverse_index,
        am_scores.T,
        ctc_scores.T,
        lm_scores.T,
        rescores.T,
        wers,
        hyps,
        refs,
    )


def createCorrptFlag(rerank_hyp, top_hyp, ref):
    if top_hyp == ref:
        if top_hyp == rerank_hyp:
            return "Same"
        else:
            return "Totally Corrupt"

    else:
        if rerank_hyp == "ref":
            return "Totally Improve"

        rerank_wer = wer(ref, rerank_hyp)
        top_wer = wer(ref, top_hyp)

        if rerank_wer > top_wer:
            return "Partial Corrupt"
        elif rerank_wer == top_wer:
            return "Neutral"
        else:
            return "Partial Improve"


def get_result(
    index_dict,
    am_scores,
    ctc_scores,
    lm_scores,
    rescores,
    wers,
    hyps,
    refs,
    am_weight,
    ctc_weight,
    lm_weight,
    rescore_weight,
):
    c = np.int64(0.0)
    s = np.int64(0.0)
    d = np.int64(0.0)
    i = np.int64(0.0)

    am_weight = np.around(np.float64(am_weight), 2)
    lm_weight = np.around(np.float64(lm_weight), 2)
    ctc_weight = np.around(np.float64(ctc_weight), 2)
    rescore_weight = np.around(np.float64(rescore_weight), 2)

    result_dict = []

    print("\n========================= Get Result ===============================\n")
    print(
        f"get result weight:\n am = {am_weight},\n ctc = {ctc_weight}, \n lm = {lm_weight}, \n rescore = {rescore_weight}"
    )

    total_score = (
        am_weight * am_scores
        + ctc_weight * ctc_scores
        + lm_weight * lm_scores
        + rescore_weight * rescores
    )

    total_score[np.isnan(total_score)] = np.NINF

    max_index = np.argmax(total_score, axis=-1)

    for utt, index in enumerate(max_index):
        c += wers[utt][index][1]
        s += wers[utt][index][2]
        d += wers[utt][index][3]
        i += wers[utt][index][4]

        if hyps[utt][0] == refs[utt]:
            if hyps[utt][index] != refs[utt]:
                corrupt_flag = "Totally Corrupt"
            else:
                corrupt_flag = "Remain Correct"
        else:
            # print(f"hyp:{type(hyps)}")
            # print(f"hyps:{hyps[utt]}")
            # print(f"top_hyp:{hyps[utt][0]}")
            # print(f"rescore_hyp:{hyps[utt][index]}")
            # print(f"ref:{refs[utt]}")

            top_wer = wer(refs[utt], hyps[utt][0])
            rerank_wer = wer(refs[utt], hyps[utt][index])
            if hyps[utt][index] == refs[utt]:
                corrupt_flag = "Totally Improve"

            elif top_wer < rerank_wer:
                corrupt_flag = "Partial Corrupt"
            elif top_wer == rerank_wer:
                corrupt_flag = "Remain Error"
            else:
                corrupt_flag = "Partial Improve"

        ref_rescore = (
            rescores[utt][hyps[utt].index(refs[utt])].item()
            if (refs[utt] in hyps[utt])
            else "Not in candidates"
        )
        ref_total = (
            total_score[utt][hyps[utt].index(refs[utt])].item()
            if (refs[utt] in hyps[utt])
            else "Not in candidates"
        )

        top_rescore = rescores[utt][0].item()
        top_total = total_score[utt][0].item()
        ans_rescore = rescores[utt][index].item()
        ans_total = total_score[utt][index].item()

        hyp_wers = [num[0] for num in wers[utt]]

        min_index = np.argmin(hyp_wers)
        index_before_min = None
        hyp_before_min = None
        score_before_min = None

        if min_index != index:
            score_index = np.argsort(total_score[utt])[::-1].tolist()
            index_before_min = score_index[: score_index.index(min_index) + 1]

        if index_before_min is not None:
            hyp_before_min = [hyps[utt][prev] for prev in index_before_min]
            score_before_min = [
                total_score[utt][prev].item() for prev in index_before_min
            ]

        result_dict.append(
            {
                # 'ASR_utt_name': index_dict[utt],
                "top_hyps": hyps[utt][0],
                "rescore_hyps": hyps[utt][index],
                "ref": refs[utt],
                "check_1": "Correct" if hyps[utt][index] == refs[utt] else "Error",
                "check_2": corrupt_flag,
                "rescores": {
                    "ref": ref_rescore,
                    "top": top_rescore,
                    "rescore": ans_rescore,
                },
                "Total_Scores": {
                    "ref": ref_total,
                    "top": top_total,
                    "rescore": ans_total,
                },
                # "Whole rescores": rescores[utt].tolist(),
                # "Whole total scores": total_score[utt].tolist(),
                "Oracle Hypothesis": hyps[utt][min_index],
                "Hyps before min wer": hyp_before_min,
                "Score before min wer": score_before_min,
            }
        )

    print(f"Result c:{c}, Result s:{s}, Result d:{d}, Result i:{i}")
    cer = (s + d + i) / (c + s + d)

    return cer, result_dict


# @jit()
def calculate_cer(
    am_scores,
    ctc_scores,
    lm_scores,
    rescores,
    wers,
    am_range=np.array([0, 10]),
    ctc_range=np.array([0, 10]),
    lm_range=np.array([0, 10]),
    rescore_range=np.array([0, 10]),
    search_step=0.1,
    min_cer=100,
    best_am=0.0,
    best_ctc=0.0,
    best_lm=0.0,
    best_rescore=0.0,
    first_flag=True,
    recog_mode=True,
):

    if first_flag:
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

    if recog_mode:
        print(f"\nsearch_step:{search_step}")
        print(f"am_range:{am_lower}, {am_upper}")
        print(f"ctc_range:{ctc_lower}, {ctc_upper}")
        print(f"lm_range:{lm_lower}, {lm_upper}")
        print(f"rescore_range:{rescore_lower}, {rescore_upper}")
        print(f"first_flag:{first_flag}")
        print(f"min cer:{min_cer}")

    for am_weight in np.arange(am_lower, am_upper + search_step, step=search_step):
        for ctc_weight in np.arange(
            ctc_lower, ctc_upper + search_step, step=search_step
        ):
            for lm_weight in np.arange(
                lm_lower, lm_upper + search_step, step=search_step
            ):
                for rescore_weight in np.arange(
                    rescore_lower, rescore_upper + search_step, step=search_step
                ):

                    c = np.int64(0.0)
                    s = np.int64(0.0)
                    d = np.int64(0.0)
                    i = np.int64(0.0)

                    am_weight = np.around(am_weight, 2)
                    ctc_weight = np.around(ctc_weight, 2)
                    lm_weight = np.around(lm_weight, 2)
                    rescore_weight = np.around(rescore_weight, 2)

                    total_score = (
                        am_weight * am_scores
                        + ctc_weight * ctc_scores
                        + lm_weight * lm_scores
                        + rescore_weight * rescores
                    )

                    total_score[np.isnan(total_score)] = np.NINF
                    total_score[np.isposinf(total_score)] = np.NINF

                    max_index = np.argmax(total_score, axis=-1)

                    for utt, index in enumerate(max_index):
                        if utt >= len(wers) or index >= len(wers[utt]):
                            print(f"total_score:{total_score[utt]}")
                            print(f"total_score:{total_score[utt][index]}")
                            print(f"max_index: {max_index[utt]}")
                        c += wers[utt][index][1]
                        s += wers[utt][index][2]
                        d += wers[utt][index][3]
                        i += wers[utt][index][4]

                    cer = (s + d + i) / (c + s + d)

                    # print(f'weight = {[am_weight, ctc_weight, lm_weight, rescore_weight]}, cer = {cer}')
                    if min_cer > cer:
                        best_am = am_weight
                        best_ctc = ctc_weight
                        best_lm = lm_weight
                        best_rescore = rescore_weight

                        min_cer = cer
                        if recog_mode:
                            print(f"Old cer:{min_cer}")
                            print(
                                f"New am:{best_am}, New ctc:{best_ctc}, New lm:{best_lm}, New rescore:{best_rescore}"
                            )
                            print(f"New cer:{cer}\n")

    if search_step <= 0.01:
        return best_am, best_ctc, best_lm, best_rescore, min_cer

    elif search_step <= 0.05:
        return calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            search_step=0.01,
            min_cer=min_cer,
            best_am=best_am,
            best_ctc=best_ctc,
            best_lm=best_lm,
            best_rescore=best_rescore,
            first_flag=False,
            recog_mode=recog_mode,
        )

    elif search_step <= 0.1:
        return calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            search_step=0.05,
            min_cer=min_cer,
            best_am=best_am,
            best_ctc=best_ctc,
            best_lm=best_lm,
            best_rescore=best_rescore,
            first_flag=False,
            recog_mode=recog_mode,
        )
    else:  # search_step = 0.2
        return calculate_cer(
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
            search_step=search_step * 0.5,
            min_cer=min_cer,
            best_am=best_am,
            best_ctc=best_ctc,
            best_lm=best_lm,
            best_rescore=best_rescore,
            first_flag=False,
            recog_mode=recog_mode,
        )


def calculate_cerOnRank(am_scores, ctc_scores, lm_scores, rescores, wers, withLM=False):
    c = np.int64(0.0)
    s = np.int64(0.0)
    d = np.int64(0.0)
    i = np.int64(0.0)

    # Here, the rescores will be the rank score

    am_rank = (-am_scores).argsort(axis=-1, kind="stable")
    ctc_rank = (-ctc_scores).argsort(axis=-1, kind="stable")
    lm_rank = (-lm_scores).argsort(axis=-1, kind="stable")

    am_rank = np.reciprocal((am_rank + 1).astype("float32"))
    ctc_rank = np.reciprocal((ctc_rank + 1).astype("float32"))
    lm_rank = np.reciprocal((lm_rank + 1).astype("float32"))

    if withLM:
        total_rank = am_rank + ctc_rank + lm_rank + rescores
    else:
        total_rank = am_rank + ctc_rank + rescores

    max_index = total_rank.argmax(axis=-1)

    for utt, index in enumerate(max_index):
        c += wers[utt][index][1]
        s += wers[utt][index][2]
        d += wers[utt][index][3]
        i += wers[utt][index][4]

    cer = (s + d + i) / (c + s + d)

    return cer


def get_resultOnRank(
    index_dict,
    am_scores,
    ctc_scores,
    lm_scores,
    rescores,
    wers,
    hyps,
    refs,
):
    c = np.int64(0.0)
    s = np.int64(0.0)
    d = np.int64(0.0)
    i = np.int64(0.0)

    result_dict = []
    # Here, the rescores will be the rank score
    am_rank = (-am_scores).argsort(dim=-1)
    ctc_rank = (-ctc_scores).argsort(dim=-1)
    lm_rank = (-lm_scores).argsort(dim=-1)

    am_rank = np.reciprocal(am_rank + 1)
    ctc_rank = np.reciprocal(ctc_rank + 1)
    lm_rank = np.reciprocal(lm_rank + 1)

    total_rank = am_rank + ctc_rank + lm_rank + rescores

    max_index = torch.argmax(total_rank, dim=-1)

    for utt, index in enumerate(max_index):
        if hyps[utt][0] == refs[utt]:
            if hyps[utt][index] != refs[utt]:
                corrupt_flag = "Totally Corrupt"
            else:
                corrupt_flag = "Remain Correct"
        else:

            top_wer = wer(refs[utt], hyps[utt][0])
            rerank_wer = wer(refs[utt], hyps[utt][index])
            if hyps[utt][index] == refs[utt]:
                corrupt_flag = "Totally Improve"

            elif top_wer < rerank_wer:
                corrupt_flag = "Partial Corrupt"
            elif top_wer == rerank_wer:
                corrupt_flag = "Remain Error"
            else:
                corrupt_flag = "Partial Improve"

        c += wers[utt][max_index][1]
        s += wers[utt][max_index][2]
        d += wers[utt][max_index][3]
        i += wers[utt][max_index][4]

        result_dict.append(
            {
                "ASR_utt_name": index_dict[utt],
                "top_hyps": hyps[utt][0],
                "rescore_hyps": hyps[utt][index],
                "ref": refs[utt],
                "check_1": "Correct" if hyps[utt][index] == refs[utt] else "Error",
                "check_2": corrupt_flag,
            }
        )

    cer = (s + d + i) / (c + s + d)

    return cer, result_dict


def get_result_simp(scores, rescores, wers, alpha, beta):
    c = np.int64(0.0)
    s = np.int64(0.0)
    d = np.int64(0.0)
    i = np.int64(0.0)

    alpha = np.float64(alpha)
    beta = np.float64(beta)

    result_dict = list()

    for score, rescore, wer in zip(scores, rescores, wers):
        total_score = alpha * score + beta * rescore

        max_index = np.argmax(total_score)

        c += wer[max_index][1]
        s += wer[max_index][2]
        d += wer[max_index][3]
        i += wer[max_index][4]

    cer = (s + d + i) / (c + s + d)

    return c, s, d, i, cer


# @jit()
def calculate_cer_simp(
    scores,
    rescores,
    wers,
    alpha_range=[0, 10],
    beta_range=[0, 10],
    search_step=0.1,
    cer=100,
):
    min_cer = np.float64(cer)

    print(f"search_step = {search_step}")

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

    print(f"alpha:{alpha_range}")
    print(f"beta: {beta_range}")

    for alpha in np.arange(alpha_range[0], alpha_range[1] + 0.01, step=search_step):
        for beta in np.arange(beta_range[0], beta_range[1] + 0.01, step=search_step):
            c = np.int64(0.0)
            s = np.int64(0.0)
            d = np.int64(0.0)
            i = np.int64(0.0)
            for score, rescore, wer in zip(scores, rescores, wers):
                total_score = alpha * score + beta * rescore

                max_index = np.argmax(total_score)

                c += wer[max_index][1]
                s += wer[max_index][2]
                d += wer[max_index][3]
                i += wer[max_index][4]
            cer = (s + d + i) / (c + s + d)

            if min_cer > cer:
                best_alpha = alpha.copy()
                best_beta = beta.copy()

                print(f"alpha:{alpha}, beta:{beta}, cer:{cer}")

                alpha_lower = (alpha - search_step) if alpha - search_step >= 0 else 0
                alpha_upper = alpha + search_step

                beta_lower = (beta - search_step) if (beta - search_step >= 0) else 0
                beta_upper = beta + search_step

                min_cer = cer

    print(f"best_alpha:{best_alpha}, best_beta:{best_beta}, min_cer:{min_cer}")

    if search_step <= 0.01:
        return best_alpha, best_beta, min_cer
    elif search_step <= 0.05:
        return calculate_cer_simp(
            scores,
            rescores,
            wers,
            alpha_range=[alpha_lower, alpha_upper],
            beta_range=[beta_lower, beta_upper],
            search_step=0.01,
            cer=min_cer,
        )
    elif search_step <= 0.1:
        return calculate_cer_simp(
            scores,
            rescores,
            wers,
            alpha_range=[alpha_lower, alpha_upper],
            beta_range=[beta_lower, beta_upper],
            search_step=0.05,
            cer=min_cer,
        )
    else:
        return calculate_cer_simp(
            scores,
            rescores,
            wers,
            alpha_range=[alpha_lower, alpha_upper],
            beta_range=[beta_lower, beta_upper],
            search_step=search_step * 0.5,
            cer=min_cer,
        )


def prepareRescoreDict(data_json):
    name_dict = dict()
    inverse_dict = dict()

    hyps_dict = []
    wers = dict()

    scores = []
    rescores = []
    if isinstance(data_json, list):
        for i, data in enumerate(data_json):
            name_dict[data["name"]] = i
            inverse_dict[i] = data["name"]

            wers[i] = data["err"]

            hyps_dict.append(data["hyps"])

            scores.append(data["score"])
            rescores.append([0.0 for _ in data["score"]])

    elif isinstance(data_json, dict):
        for i, key in enumerate(data_json.keys()):
            name_dict[key] = i
            inverse_dict[i] = key

            wers[i] = data_json[key]["err"]

            hyps_dict.append(data_json[key]["hyps"])

            scores.append(data_json[key]["score"])
            rescores.append([0.0 for _ in data_json[key]["score"]])

    scores = np.array(list(zip_longest(*scores, fillvalue=np.NINF)), dtype=np.float32)
    rescores = np.array(
        list(zip_longest(*rescores, fillvalue=np.NINF)), dtype=np.float32
    )

    return name_dict, inverse_dict, hyps_dict, scores.T, rescores.T, wers

import numpy as np

def prepare_score_dict(data_json):
    index_dict = dict()

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

        simp_flag = False
        if ('score' in data_json[0].keys):
            scores = []

            simp_flag = True
        
        else:
            am_scores = []
            lm_scores = []
            ctc_scores = []
        rescores = []
        wers = []
        for i, data in data_json:
            index_dict[data['name']] = i

            


    elif (isinstance(data_json, dict)):
        """
        if data_json is a dict, the format should be
        name:{
            hyps:[],
            am_score:[],
            ...
        }
        """
         
        pass
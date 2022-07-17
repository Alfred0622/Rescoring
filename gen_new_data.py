import json
dataset = 'csj'
recog_set = ['dev', 'eval1', 'eval2', 'eval3']
# setting = ['withLM']
setting = ['noLM']
# setting = ['withLM', 'noLM']
nbest = 50

for s in setting:
    for task in recog_set:
        nbest_err = []
        print(f'{s}:{task}')
        with open(f'./data/{dataset}/{s}/{task}_result.txt', 'r') as f:
            temp_err = []
            for i, line in enumerate(f):
                if ('Scores' in line):
                    scores = line.split()[-4:]
                    temp_err.append(scores)
                    if (len(temp_err) == nbest):
                        nbest_err.append(temp_err)
                        temp_err = []
            names = []
            nbest_token = []
            nbest_score = []
            ref_text = []
            with open(f'./data/{dataset}/{s}/{task}_data.json', 'r') as f:
                j = json.load(f)
                for k in j['utts'].keys():
                    token = []
                    score = []
                    for i, h in enumerate(j['utts'][k]['output']):
                        token.append(h["rec_text"])
                        score.append(h['score'])
                    nbest_token.append(token)
                    nbest_score.append(score)
                    ref_text.append(j['utts'][k]['output'][0]['text'])
                    names.append(k)

            dataset = []
            for name, token, score, err, ref in zip(names, nbest_token, nbest_score, nbest_err, ref_text):
                temp_dict = dict()
                temp_dict['name'] = name
                temp_dict['token'] = [t[:-5]  for  t  in token]
                temp_dict['score'] = score
                temp_dict['err'] =  [[int(s) for s in sc] for sc in err]
                temp_dict['ref'] =  ref 
                dataset.append(temp_dict)
            print(len(dataset))
        with open(f"./data/{dataset}/{task}/data_{s}.json", 'w') as f:
            json.dump(dataset, f, ensure_ascii=False, indent = 4)
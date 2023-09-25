def getBertPretrainName(dataset):
    if (dataset in ['aishell', 'aishell2']):
        return 'bert-base-chinese'
    elif (dataset in ['tedlium2', 'librispeech']):
        return 'bert-base-uncased'
    elif (dataset in ['csj']):
        return 'cl-tohoku/bert-base-japanese'

def getGPTPretrainName(dataset):
    if (dataset in ['aishell', 'aishell2']):
        return 'bert-base-chinese'
    elif (dataset in ['tedlium2', 'librispeech']):
        return 'gpt2'
    elif (dataset in ['csj']):
        return 'ClassCat/gpt2-base-japanese-v2'

def getBartPretrainName(dataset):
    if (dataset in ['aishell', 'aishell2']):
        return 'fnlp/bart-base-chinese'
    elif (dataset in ['tedlium2', 'librispeech']):
        return 'facebook/bart-base'
    elif (dataset in ['csj']):
        return 'ku-nlp/bart-base-japanese'
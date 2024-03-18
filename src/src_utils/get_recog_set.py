def get_valid_set(dataset):
    
    if (dataset in ['aishell2']):
        return 'dev_ios'
    elif (dataset in ["librispeech"]):
        return 'dev'
    else:
        return 'dev'

def get_recog_set(dataset):

    if (dataset in ['aishell', 'tedlium2']):
        return ['dev', 'test']
    elif (dataset in ['aishell2']):
        return ['dev', 'test_mic', 'test_ios', 'test_android']
    elif (dataset in ['csj']):
        return ['dev', 'eval1', 'eval2', 'eval3']
    elif (dataset in ['librispeech']):
        return ['dev', 'dev_clean', 'dev_other', 'test_clean', 'test_other']
    else:
        raise ValueError(f"Dataset {dataset} not implemented")

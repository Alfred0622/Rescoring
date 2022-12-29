def get_recog_set(dataset):

    if (dataset in ['aishell', 'tedlium2']):
        return ['dev', 'test']
    elif (dataset in ['aishell2']):
        return ['test_mic', 'test_iOS', 'test_android']
    elif (dataset in ['csj']):
        return ['dev', 'eval1', 'eval2', 'eval3']
    elif (dataset in ['librispeech']):
        return ['dev_clean', 'dev_other', 'test_clean', 'test_other']
    else:
        logging.warning(f"Dataset {dataset} not implemented")
        return None

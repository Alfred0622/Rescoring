def get_vocab(dict_file):
    vocab_dict = list()
    vocab_dict.append("<pad>")
    for i, line in enumerate(dict_file):
        token, idx = line.strip().split(' ')
        vocab_dict.append(token)
    vocab_dict.append("<eos>")
    
    return vocab_dict
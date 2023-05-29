def change_unicode(token_list):
    for i, token in  enumerate(token_list):
        if (65281 <= ord(token) <= 65374 ): # if 全形
            token_list[i] = chr(ord(token) - 65248)
    
    return token_list

def preprocess_string(string, dataset):
    string = string.replace("<eos>", "").strip().split()
    string = [token for token in string]

    if (dataset in ['csj']):
        string = change_unicode(string)
        string = "".join(string)
    else:
        string = " ".join(string)
    
    return string
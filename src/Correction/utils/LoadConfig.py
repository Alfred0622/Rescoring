import yaml
import json

def load_config(file_name):
    with open(file_name) as f:
        conf = yaml.load(f.read(), Loader=yaml.FullLoader)
        args = conf['args']
        train_args = conf['train']
        # train_args include epoch, batch_size, lr
        recog_args = conf["recog"]
        
        if ('adapt' in conf.keys()):
            adapt_args = conf['conf']
            return args, adapt_args, train_args, recog_args
        
        return args, train_args, recog_args
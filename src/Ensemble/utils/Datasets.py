import torch
from torch.utils.data import Dataset

class ensembleDataset(Dataset):
    def __init__(self, data_dict):
        """
        data_dict: dict
        dict architecture:
        {
            name: {
                    CLM: score,
                    MLM: score,

                  }
        }
        """
        dataset_list = list()
        feature_vec = torch.tensor([], dtype = torch.float32)
        
        for name in data_dict.keys():
            pass

def prepare_emsemble_dataset(data_json, data_dict):
    for data in data_json:
        if (not data['name'] in data_dict.keys()):
            data_dict[data['name']] = []
        
        for key in data.keys():
            if (key in ['name']):
                continue
            data_dict[data['name']].append(data['rescores'])
        
    return data_dict
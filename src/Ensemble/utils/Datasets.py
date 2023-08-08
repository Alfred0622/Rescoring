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
import torch
from torch.utils.data import Dataset

class concatDataset(Dataset):
    # Dataset for BertComparision
    # Here, data will have
    # 1.  [CLS] seq1 [SEP] seq2
    # 2.  labels
    def __init__(self, nbest_list, nbest=10):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list

    def __getitem__(self, idx):
        return (
            self.data[idx]["token"],
            self.data[idx]['label']
        )

    def __len__(self):
        return len(self.data)

class compareRecogDataset(Dataset):
    def __init__(self, nbest_list, nbest=10):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list

    def __getitem__(self, idx):
        return (
            self.data[idx]['name'],
            self.data[idx]["token"],
            self.data[idx]['pair'],
            self.data[idx]['text'],
            self.data[idx]['score'],
            self.data[idx]['err'],
            self.data[idx]['ref'],
        )

    def __len__(self):
        return len(self.data)

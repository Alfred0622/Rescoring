import torch
from torch.utils.data import Dataset, DataLoader

class correctDataset(Dataset):
    def __init__(self, nbest_list, nbest=10):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list
        self.nbest = nbest

    def __getitem__(self, idx):
        return (
            self.data[idx]["token"][: self.nbest],
            self.data[idx]["ref_token"][1:],
            self.data[idx]['err'][: self.nbest],
            self.data[idx]['text'][: self.nbest],
            self.data[idx]['ref'],
            self.data[idx]['score']
        )

    def __len__(self):
        return len(self.data)

class correctRecogDataset(Dataset):
    def __init__(self, nbest_list, nbest=10):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list
        self.nbest = nbest

    def __getitem__(self, idx):
        return (
            self.data[idx]["token"][0],
            self.data[idx]["ref_token"],
            self.data[idx]['text'][: self.nbest],
            self.data[idx]['ref'],
        )
    def __len__(self):
        return len(self.data)


class correctDataset_withPho(Dataset):
    def __init__(self, nbest_list):
        self.data = nbest_list

    def __getitem__(self, idx):
        return (
            self.data[idx]["token"],
            self.data[idx]["phoneme"],
            self.data[idx]["ref_token"],
        )

    def __len__(self):
        return len(self.data)

class nBestAlignDataset(Dataset):
    def __init__(self, nbest_list):
        """
        nbest_dict: {token seq : CER}
        """
        self.data = nbest_list

    def __getitem__(self, idx):
        return (
            self.data[idx]["token"],
            self.data[idx]["ref_token"],  # avoid [CLS]
            self.data[idx]["ref"],
        )

    def __len__(self):
        return len(self.data)

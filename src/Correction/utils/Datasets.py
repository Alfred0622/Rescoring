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
            self.data["token"][idx],
            self.data["ref_token"][idx][1:],
        )

    def __len__(self):
        return len(self.data['token'])

class correctRecogDataset(Dataset):
    def __init__(self, nbest_list, nbest=10):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list
        self.nbest = nbest

    def __getitem__(self, idx):
        return (
            self.data["token"][idx],
            self.data['ref'][idx],
        )
    def __len__(self):
        return len(self.data['token'])


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
            self.data[idx]["ref_token"][1:],  # avoid [CLS]
            self.data[idx]["ref"],
        )

    def __len__(self):
        return len(self.data)

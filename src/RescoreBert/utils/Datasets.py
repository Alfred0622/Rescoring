import torch
from torch.utils.data import Dataset, DataLoader

class adaptionDataset(Dataset):
    # Use only for domain adaption in MLM bert
    # Only use groundtruth
    def __init__(self, nbest_list):
        self.data = nbest_list

    def __getitem__(self, idx):
        return self.data[idx]["ref_token"]

    def __len__(self):
        return len(self.data)

class pllDataset(Dataset):
    # Training dataset
    def __init__(self, nbest_list, nbest = 50):
        """
        nbest_list: list()
        """
        self.data = nbest_list
        self.nbest = nbest

    def __getitem__(self, idx):
        return (
            self.data[idx]["token"][: self.nbest],
            self.data[idx]["text"][: self.nbest],
            self.data[idx]["score"][: self.nbest],
            self.data[idx]["err"][: self.nbest],
            self.data[idx]["pll"][: self.nbest],
        )
        #    self.data[idx]['name'],\

    def __len__(self):
        return len(self.data)

class nBestDataset(Dataset):
    def __init__(self, nbest_list, nbest = 50):
        """
        nbest_list: list()
        """
        self.data = nbest_list
        self.nbest = nbest

    def __getitem__(self, idx):
        return (
            self.data[idx]["token"][: self.nbest],
            self.data[idx]["text"][: self.nbest],
            self.data[idx]["score"][: self.nbest],
            self.data[idx]["err"][: self.nbest],
        )
        #    self.data[idx]['name'],\
    def __len__(self):
        return len(self.data)

class rescoreDataset(Dataset):
    def __init__(self, nbest_list, nbest=10):
        """
        nbest_list: list() of dict()
        """
        self.data = nbest_list
        self.nbest = nbest

    def __getitem__(self, idx):
        return (
            self.data[idx]["name"],
            self.data[idx]["token"][: self.nbest],
            self.data[idx]["text"][: self.nbest],
            self.data[idx]["score"][: self.nbest],
            self.data[idx]["ref"],
            self.data[idx]["err"][: self.nbest],
        )

    def __len__(self):
        return len(self.data)
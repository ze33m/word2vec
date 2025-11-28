from torch.utils.data import Dataset
import torch
from typing import Counter
import numpy as np
from tqdm import tqdm
import bisect
import time

class NegativeSamplingDataset(Dataset):
    def __init__(self, docs, window_size, negatives_number):
        self.docs = docs
        self.negatives_number = negatives_number
        self.window_size = window_size
        self.make_dataset()

    def make_dataset(self):
        targets = torch.tensor(data=[],dtype=torch.long)
        contexts = torch.tensor(data=[],dtype=torch.long)
        for doc in self.docs:
            for i in range(self.window_size + 1, len(doc) - self.window_size):
                     pass
       

    def __len__(self):
        pass
    
    def __getitem__(self, i):
        pass
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
        pass
       

    def __len__(self):
        return self.cum_pairs[-1] #cum t-shirt
    
    def __getitem__(self, i):
        doc_idx = bisect.bisect_right(self.cum_pairs, i) - 1
        offset_in_doc = i - self.cum_pairs[doc_idx]

        doc_id, valid_count = self.valid_pos[doc_idx]
        tokens = self.docs[doc_id]

        pair_in_doc = offset_in_doc//(self.window_size * 2)
        context_offset = offset_in_doc%(self.window_size * 2)

        target_pos = self.window_size + pair_in_doc
        target_id = tokens[target_pos]

        left = target_pos - self.window_size
        right = target_pos + self.window_size
        contexts = list(range(left, right + 1))
        contexts.remove(target_pos)

        context_pos = contexts[context_offset]
        context_id = tokens[context_pos]
        return torch.tensor(target_id, dtype=torch.long), torch.tensor(context_id, dtype=torch.long)
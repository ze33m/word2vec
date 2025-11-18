from torch.utils.data import Dataset
import torch
from typing import Counter
import numpy as np
from tqdm import tqdm
import bisect

class NegativeSamplingDataset(Dataset):
    def __init__(self, docs, window_size, negatives_number):
        self.docs = docs
        self.negatives_number = negatives_number
        self.window_size = window_size
        self.make_dataset()

    def make_dataset(self):
        self.valid_pos = []
        self.vocab = set()
        token_freqs = Counter()
        for doc_idx, tokens in enumerate(tqdm(self.docs, desc='counting pairs')):
            self.vocab.update(tokens)
            token_freqs.update(tokens)

            if len(tokens) > 2 * self.window_size:
                valid_count = len(tokens) - 2 * self.window_size
                self.valid_pos.append((doc_idx, valid_count))

        self.cum_pairs = [0]

        for doc_idx, valid_count in self.valid_pos:
            self.cum_pairs.append(self.cum_pairs[-1] + valid_count * 2 * self.window_size)

        self.vocab = list(self.vocab)
        self.wtoi = {word: i for i, word in enumerate(self.vocab)}
        self.itow = {i: word for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        self.word_probs = np.array(
            [token_freqs[w]**0.75 for w in self.vocab], dtype=np.float32
        )
        self.word_probs /= self.word_probs.sum()


    def __len__(self):
        return self.cum_pairs[-1]
    
    def __getitem__(self, i):
        doc_idx = bisect.bisect_right(self.cum_pairs, i) - 1
        offset_in_doc = i - self.cum_pairs[doc_idx]

        doc_id, valid_count = self.valid_pos[doc_idx]
        tokens = self.docs[doc_id]

        pair_in_doc = offset_in_doc//(self.window_size * 2)
        context_offset = offset_in_doc%(self.window_size * 2)

        target_pos = self.window_size + pair_in_doc
        target_word = tokens[target_pos]
        target_id = self.wtoi[target_word]

        left = target_pos - self.window_size
        right = target_pos + self.window_size
        contexts = list(range(left, right + 1))
        contexts.remove(target_pos)

        context_pos = contexts[context_offset]
        context_word = tokens[context_pos]
        context_id = self.wtoi[context_word]

        return torch.tensor(target_id, dtype=torch.long), torch.tensor(context_id, dtype=torch.long)
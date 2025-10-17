from torch.utils.data import Dataset
import torch
from typing import Counter
import numpy as np
import re
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

class NegativeSamplingDataset(Dataset):
    def __init__(self, docs, window_size, negatives_number):
        self.preprocess_docs(docs)
        self.negatives_number = negatives_number
        self.window_size = window_size
        self.converts()
        self.make_dataset()
        self.calc_word_probs()

    def preprocess_docs(self, docs):
        self.stop_words = set(stopwords.words('russian'))
        self.stemmer = SnowballStemmer("russian")
        self.docs = []
        print('Предобработка корпуса')
        for i in tqdm(range(len(docs))):
            s = docs[i]
            self.docs.append(self.preprocessing(s))

    def preprocessing(self, raw_text:str):
        raw_text = raw_text.lower()
        raw_text = re.sub(r'[^\w\s]|\d', '', raw_text) 
        raw_text = raw_text.replace('\n', ' ')
        raw_text = re.sub(r'[\s]+', ' ', raw_text)
        tokens = [self.stemmer.stem(i) for i in raw_text.split() if i not in self.stop_words]
        return tokens
    
    def converts(self):
        self.tokens = []
        for tokens in self.docs:
            self.tokens += tokens
        self.vocab = sorted(set(self.tokens))
        self.wtoi = {word: i for i, word in enumerate(self.vocab)}
        self.itow = {i: word for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def make_dataset(self):
        targets, contexts = [], []
        for tokens in self.docs:
            for i in range(self.window_size, len(tokens) - self.window_size):
                target = self.wtoi[tokens[i]]
                context = [self.wtoi[j] for j in tokens[i - self.window_size : i + self.window_size + 1]]
                context.remove(target)
                for j in range(len(context)):
                    targets.append(target)
                    contexts.append(context[j])
        self.targets = torch.LongTensor(targets)
        self.contexts = torch.LongTensor(contexts)

    def calc_word_probs(self):
        freqs = Counter(self.tokens)
        self.word_probs = np.array(
            [freqs[w]**0.75 for w in self.vocab], dtype=np.float32
        )
        self.word_probs /= self.word_probs.sum()

    def get_random_negatives(self):
        return torch.LongTensor([self.wtoi[i] for i in np.random.choice(self.vocab, size=self.negatives_number, p=self.word_probs)])
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, i):
        target = self.targets[i]
        context = self.contexts[i]
        negatives = self.get_random_negatives()
        return target, context, negatives
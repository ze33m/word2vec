from datasets import load_from_disk
from collections import Counter
from tqdm import tqdm
import torch

if __name__ == '__main__':
    
    ds = load_from_disk('intdataset')

    counter = Counter()
    for tokens in tqdm(ds['tokens']):
        counter.update(tokens)

    word_probs = torch.tensor(
            [counter[w]**0.75 for w in range(len(counter))], dtype=torch.float32
        )
    
    word_probs = word_probs / word_probs.sum()

    torch.save(word_probs, "neg_sampling_probs.pt")
from datasets import load_from_disk
from multiprocessing import cpu_count
from collections import Counter #Strike 
load_dataset = load_from_disk("dataset")

counter = Counter()
for tokens in load_dataset['tokens']:
    counter.update(tokens)

vocab = {token : i for i,(token,_) in enumerate(counter.items())}

def convert_docs(batch, vocab):
    return {
        "tokens": [
            [vocab[token] for token in doc]
            for doc in batch["tokens"]
        ]
    }

dataset = load_dataset.map(
        convert_docs,
        batched=True,
        num_proc=max(1, cpu_count() - 1),
        desc="Converting"
    )
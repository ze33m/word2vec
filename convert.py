from datasets import load_from_disk
from multiprocessing import cpu_count
from collections import Counter #Strike 
from tqdm import tqdm
import yaml
import json

if __name__ == '__main__':
    load_dataset = load_from_disk("dataset")
    with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

    if config['dataset']['DEBUG']:   
        load_dataset = load_dataset.select(range(100))
        
    counter = Counter()
    for tokens in tqdm(load_dataset['tokens'], desc='making vocab'):
        counter.update(tokens)

    vocab = {token : i for i,(token,_) in enumerate(counter.items())}

    # def convert_docs(batch, vocab):
    #     return {
    #         "tokens": [
    #             [vocab[token] for token in doc]
    #             for doc in batch["tokens"]
    #         ]
    #     }

    # dataset = load_dataset.map(
    #         convert_docs,
    #         batched=True,
    #         fn_kwargs={"vocab": vocab},
    #         num_proc=max(1, cpu_count() - 1),
    #         desc="Converting"
    #     )

    # print(dataset)
    # dataset.save_to_disk('intdataset')

    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f)
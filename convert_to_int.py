from datasets import load_from_disk
from multiprocessing import cpu_count
from collections import Counter #Strike 
from tqdm import tqdm
import yaml
import json
"""
dataset/   --->   intdataset/

Преобразование токенов в int + vocab
"""

if __name__ == '__main__':
    load_dataset = load_from_disk("dataset")
    with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

    if config['dataset']['DEBUG']:   
        load_dataset = load_dataset.select(range(100))
    
    # считаем количество вхождений всех слов 
    counter = Counter()
    for tokens in tqdm(load_dataset['tokens']):
        counter.update(tokens)

    min_count = config["dataset"].get("min_count", 5)
    
    # в словарь добавляем только те, что встречаюстя хотя бы min_count раз
    items = [(tok, cnt) for tok, cnt in counter.items() if cnt >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    vocab = {"<unk>" : 0}
    for i, (token,_) in enumerate(items, start=1):
         vocab[token] = i

    print(len(vocab))
    with open("vocab.json", "w", encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)
    
    # преобразуем токены в числа
    def convert_docs(batch, vocab):
        return {
            "tokens": [
                [vocab.get(token, 0) for token in doc]
                for doc in batch["tokens"]
            ]
        }

    # применение convert_docs к нашему датасету с распараллеливанием
    dataset = load_dataset.map(
            convert_docs,
            batched=True,
            fn_kwargs={"vocab": vocab},
            num_proc=max(1, cpu_count() - 1),
            desc="Converting"
        )

    dataset.save_to_disk('intdataset')

import yaml
from word2vec import w2v_ns
from dataset import NegativeSamplingDataset
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import pickle 
import torch
import datetime
import os
from multiprocessing import cpu_count
from datasets import load_from_disk
from functools import partial

def collate_fn(batch, dataset):
        targets = torch.stack([i[0] for i in batch])
        contexts = torch.stack([i[1] for i in batch])
        negatives = torch.multinomial(
            dataset.word_probs, 
            dataset.negatives_number * len(batch), 
            replacement=True
        ).view(len(batch), dataset.negatives_number)

        return targets, contexts, negatives

if __name__ == "__main__":

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    load_dataset = load_from_disk("intdataset")

    if config['dataset']['DEBUG']:   
        load_dataset = load_dataset.select(range(100))

    print(load_dataset)


    window_size=config['dataset']['window_size']
    negatives_number=config['dataset']['negatives_number']
    batch_size=config['dataset']['batch_size']
    embed_size=config['model']['embed_size']
    lr=config['train']['lr']
    epochs=config['train']['epochs']

    dataset = NegativeSamplingDataset(load_dataset['tokens'], window_size, negatives_number)
    print(dataset)

    
    custom_collate_fn = partial(collate_fn, dataset=dataset)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers = max(1, cpu_count() - 1))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = w2v_ns(dataset=dataset, embed_size=embed_size).to(device)

    opt = optim.Adam(model.parameters(), lr)

    def train():
        print(f'Обучение на {device}')
        for epoch in tqdm(range(epochs)):
            total_loss = 0
            for target, context, negatives in tqdm(dl):
                target = target.to(device)
                context = context.to(device)
                negatives = negatives.to(device)
                loss = model(target, context, negatives)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() / batch_size
                pass
            print(f'Epoch num: {epoch+1}, loss value: {total_loss:.3f}')

    train()
    os.makedirs("model", exist_ok=True)
    torch.save(model, 'model/model.pth')

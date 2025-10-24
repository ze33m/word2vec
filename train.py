
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
from datasets import load_from_disk

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset = load_from_disk("dataset")

if config['dataset']['DEBUG']:   
    dataset = dataset.select(range(10))

print(dataset)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

window_size=config['dataset']['window_size']
negatives_number=config['dataset']['negatives_number']
batch_size=config['dataset']['batch_size']
embed_size=config['model']['embed_size']
lr=config['train']['lr']
epochs=config['train']['epochs']



dataset = NegativeSamplingDataset(dataset['tokens'], window_size, negatives_number)
print(dataset)
dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        print(f'Epoch num: {epoch+1}, loss value: {total_loss:.3f}')


if __name__ == "__main__":
    train()
    os.makedirs("model", exist_ok=True)
    torch.save(model, 'model/model.pth')

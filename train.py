
import yaml
from debug_dataset import debug_dataset
from word2vec import w2v_ns
from dataset import NegativeSamplingDataset
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import pickle 
import torch
import datetime

def now():
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)



window_size=config['dataset']['window_size']
negatives_number=config['dataset']['negatives_number']
batch_size=config['dataset']['batch_size']
embed_size=config['model']['embed_size']
lr=config['train']['lr']
epochs=config['train']['epochs']

with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

dataset = NegativeSamplingDataset(dataset, window_size, negatives_number)

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

def save():
    with open(f'models/model{now()}.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    # train()
    torch.save(dataset.__dict__, 'dataset.pt')
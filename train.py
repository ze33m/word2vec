from datasets import load_dataset
import yaml
from debug_dataset import debug_dataset
from word2vec import w2v_ns
from preprocess import NegativeSamplingDataset
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import pickle 
import datetime

def now():
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if config['dataset']['DEBUG']:   
    dataset = debug_dataset
else:
    dataset = load_dataset('0x7o/taiga', split='train')['text']

window_size=config['dataset']['window_size']
negatives_number=config['dataset']['negatives_number']
batch_size=config['dataset']['batch_size']
embed_size=config['model']['embed_size']
lr=config['train']['lr']
epochs=config['train']['epochs']

dataset = NegativeSamplingDataset(dataset, window_size, negatives_number)

dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = w2v_ns(dataset=dataset, embed_size=embed_size)
opt = optim.Adam(model.parameters(), lr)

def train():
    print('Обучение')
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for target, context, negatives in tqdm(dl):
            loss = model(target, context, negatives)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() / batch_size
        print(f'Epoch num: {epoch+1}, loss value: {total_loss:.3f}')


if __name__ == "__main__":
    train()
    with open(f'models/model{now()}.pkl', 'wb') as file:
        pickle.dump(model, file)
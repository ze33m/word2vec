
import yaml
from word2vec import w2v_ns
from dataset import PairsStream
from torch import optim
from tqdm import tqdm
import torch
import os

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    window_size=config['dataset']['window_size']
    negatives_number=config['dataset']['negatives_number']
    batch_size=config['dataset']['batch_size']
    embed_size=config['model']['embed_size']
    lr=config['train']['lr']
    epochs=config['train']['epochs']

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = w2v_ns(embed_size=embed_size).to(device)

    opt = optim.Adam(model.parameters(), lr)
    # пока попробуем обучить на 50млн парах. посмотрим, все ли работает и обучится ли он чему 
    stream = PairsStream("pairs/shard-00000.parquet", batch_size=300_000, neg_num=5)

    def train():
        print(f'Обучение на {device}')
        for epoch in tqdm(range(epochs)):
            total_loss = 0
            for target, context, negatives in tqdm(stream):
                target = target.to(device)
                context = context.to(device)
                negatives = negatives.to(device)
                loss = model(target, context, negatives)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() / batch_size
            print(f'Epoch num: {epoch+1}, loss value: {total_loss:.3f}')

    train()
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), 'model/model.pth')
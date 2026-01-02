
import yaml
from word2vec import w2v_ns
from dataset import PairsStream
from torch import optim
from tqdm import tqdm
import torch
import os
from s3con import s3con

def get_shard(i):
    number = str(i)
    while len(number) != 5:
        number = '0' + number
    return f"pairs/shard-{number}.parquet"


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
    s3 = s3con()
    opt = optim.SparseAdam(model.parameters(), lr)
    os.makedirs("model", exist_ok=True)
    s3 = s3con()
    def train():
        print(f'Обучение на {device}')
        for i in tqdm(range(610)):
            stream = PairsStream(get_shard(i), batch_size=300_000, neg_num=5)
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
            print(f'shard num: {i+1}, loss value: {total_loss:.3f}')
            if i%100 == 0:
                torch.save(model.state_dict(), f'model/model_{i}.pth')
                s3.upload_one('taiga-model', f'model/model_{i}.pth')
    train()

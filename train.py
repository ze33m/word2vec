
import yaml
from word2vec import w2v_ns
from dataset import nsDATASET
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torch
import os
from datasets import load_from_disk


import pyarrow.parquet as pq
import torch
from tqdm import tqdm
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

class PairsStream(IterableDataset):
    def __init__(self, path, batch_size, neg_num, probs_path="neg_sampling_probs.pt"):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.neg_num = neg_num
        probs = torch.load(probs_path)
        self.sampler = torch.distributions.Categorical(probs)

    def __iter__(self):
        pf = pq.ParquetFile(self.path)
        for rb in pf.iter_batches(batch_size=self.batch_size, columns=["target", "context"]):
            # writable copy, чтобы не было warning и странных эффектов
            t = torch.from_numpy(rb.column("target").to_numpy(zero_copy_only=False).copy())
            c = torch.from_numpy(rb.column("context").to_numpy(zero_copy_only=False).copy())
            n = self.sampler.sample((t.shape[0], self.neg_num))
            yield t, c, n

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    window_size=config['dataset']['window_size']
    negatives_number=config['dataset']['negatives_number']
    batch_size=config['dataset']['batch_size']
    embed_size=config['model']['embed_size']
    lr=config['train']['lr']
    epochs=config['train']['epochs']

    dataset = nsDATASET(batch_size=int(1e6))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = w2v_ns(dataset=dataset, embed_size=embed_size).to(device)

    opt = optim.SGD(model.parameters(), lr)
    
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
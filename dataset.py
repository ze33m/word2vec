from torch.utils.data import Dataset
from datasets import load_dataset
import torch

class nsDATASET(Dataset):
    def get_shard_name(self):
        shard_id = str(self.shard_id)
        while len(shard_id) != 5:
            shard_id = '0' + shard_id
        return f'pairs/shard-{shard_id}.parquet'

    def __init__(self, shard=0, batch_size=1000, neg_num=5):
        self.shard_id = shard
        self.batch_size = batch_size
        self.neg_num = neg_num
        self.shard = load_dataset(
            "parquet",
            data_files = self.get_shard_name(),
            split="train"
        )
        word_probs = torch.load("neg_sampling_probs.pt")
        self.sampler = torch.distributions.Categorical(word_probs)
    def __len__(self):
        return len(self.shard)//self.batch_size

    def __getitem__(self, idx):
        idxs = range(idx*self.batch_size, (idx+1)*self.batch_size)
        batch = self.shard.select(idxs).with_format("torch")
        return batch["target"][:].to(torch.long), batch["context"][:].to(torch.long), self.sampler.sample((self.batch_size, self.neg_num)).to(torch.long)
# вроде все сделал. должно заработать когда появится neg_sampling_probs.pt. проверить в test.ipynb 
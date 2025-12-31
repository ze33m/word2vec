import torch
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

class PairsStream(IterableDataset):
    def __init__(self, path, batch_size, neg_num, probs_path="vocab/neg_sampling_probs.pt"):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.neg_num = neg_num
        probs = torch.load(probs_path)
        self.sampler = torch.distributions.Categorical(probs)

    def __iter__(self):
        pf = pq.ParquetFile(self.path)
        for rb in pf.iter_batches(batch_size=self.batch_size, columns=["target", "context"]):
            t = torch.from_numpy(rb.column("target").to_numpy(zero_copy_only=False).copy())
            c = torch.from_numpy(rb.column("context").to_numpy(zero_copy_only=False).copy())
            n = self.sampler.sample((t.shape[0], self.neg_num))
            yield t, c, n
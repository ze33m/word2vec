from torch import nn
import torch

class w2v_ns(nn.Module):
    def __init__(self, dataset, embed_size):
        super().__init__()
        self.dataset = dataset
        self.vocab_size = 2495767 # хардкод жоский
        self.target_emb = nn.Embedding(self.vocab_size, embed_size) 
        self.context_emb = nn.Embedding(self.vocab_size, embed_size)
        self.log_sigmoid = nn.LogSigmoid()
        
    def forward(self, target, context, negatives) -> torch.Tensor:
        target_vec = self.target_emb(target) # ( batch_size,  embed_size )
        context_vec = self.context_emb(context) # ( batch_size,  embed_size )
        negative_vecs = self.context_emb(negatives) # ( batch_size, neg_num, embed_size )

        neg_score = self.log_sigmoid(-(negative_vecs * target_vec.unsqueeze(1)).sum(dim=2)).sum(dim=1)
        pos_score = self.log_sigmoid((target_vec * context_vec).sum(dim=1))
        loss = -(neg_score + pos_score).mean()
        return loss
    
    def K_nearest(self, x_idx, k):
        
        x_vec = self.target_emb(torch.tensor(x_idx)).detach()
        all_vecs = self.target_emb.weight.detach()
        sims = nn.functional.cosine_similarity(x_vec.unsqueeze(0), all_vecs, dim=1)
        topk = torch.topk(sims, k+2)
        indices = topk.indices.tolist()
        scores = topk.values.tolist()
        result = [(i, s) for i, s in zip(indices, scores) if i!=x_idx]
        
        return result
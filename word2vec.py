from torch import nn
import torch

class w2v_ns(nn.Module):
    def __init__(self, dataset, embed_size):
        super().__init__()
        self.dataset = dataset
        self.vocab_size = dataset.vocab_size
        self.target_emb = nn.Embedding(self.vocab_size, embed_size) 
        self.context_emb = nn.Embedding(self.vocab_size, embed_size)
        self.log_sigmoid = nn.LogSigmoid()
        
    def forward(self, target, context, negatives) -> torch.Tensor:
        target_vec = self.target_emb(target) # ( batch_size,  embed_size )
        context_vec = self.context_emb(context) # ( batch_size,  embed_size )
        negative_vecs = self.context_emb(negatives) # ( batch_size, neg_num, embed_size )
        neg_score = self.log_sigmoid(-torch.bmm(negative_vecs, target_vec.unsqueeze(2))).sum(1)
        pos_score = self.log_sigmoid(torch.bmm(context_vec.unsqueeze(1), target_vec.unsqueeze(2))).squeeze(2).sum(1)
        loss = -(neg_score + pos_score).mean()
        return loss
    
    def K_nearest(self, x, k):
        x_idx = torch.tensor(self.dataset.wtoi[x], dtype=torch.long)
        x_vec = self.target_emb(x_idx).detach()
        all_vecs = self.target_emb.weight.detach()
        sims = nn.functional.cosine_similarity(x_vec.unsqueeze(0), all_vecs, dim=1)
        topk = torch.topk(sims, k+2)
        indices = topk.indices.tolist()
        scores = topk.values.tolist()
        result = [(self.dataset.itow[i], s) for i, s in zip(indices, scores) if i!=x_idx]
        
        return result
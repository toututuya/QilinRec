import numpy as np
import torch
from torch import nn

from .data import profile_arrays


class CTRModel(nn.Module):
    def __init__(self, name, vectors, codes, n_users, cfg, feature_dim):
        super().__init__()
        self.name = name
        self.register_buffer("vectors",torch.from_numpy(vectors),persistent=False)
        self.register_buffer("codes",torch.from_numpy(codes),persistent=False)
        d = cfg["id_dim"]
        self.residual = nn.Embedding(int(codes.max())+1,d)
        nn.init.normal_(self.residual.weight,std=.015)
        self.user = nn.Embedding(n_users+1,d)
        self.head = nn.Sequential(nn.Linear(d*4+feature_dim,96),nn.LayerNorm(96),nn.SiLU(),
                                  nn.Dropout(.12),nn.Linear(96,32),nn.SiLU(),nn.Linear(32,1))
        if name == "din":
            self.attention = nn.Sequential(nn.Linear(4*d,48),nn.SiLU(),nn.Linear(48,1))

    def item(self,i):
        return self.vectors[i.long()] + self.residual(self.codes[i.long()].long())

    def pool(self, target, history, mask):
        if self.name == "din":
            q = target[:,None].expand_as(history)
            local = torch.cat([q,history,q-history,q*history],dim=-1)
            scores = self.attention(local).squeeze(-1).masked_fill(~mask,-1.e9)
            weight = torch.softmax(scores,dim=1)*mask
        else:
            weight = mask / mask.sum(1,keepdim=True).clamp(min=1)
        return (history*weight[:,:,None]).sum(1)

    def forward(self, users, items, histories, x):
        target = self.item(items)
        interest = self.pool(target,self.item(histories),histories!=0)
        u = self.user(users.long())
        return self.head(torch.cat([u,target,interest,target*interest,x],dim=1)).squeeze(-1)


class RankingDataset:
    def __init__(self, rows, frame, cfg, scale, device, labeled=True, shortlist=None):
        user,hist = profile_arrays(rows,cfg["history_length"])
        to_device = lambda a: torch.as_tensor(a,device=device)
        self.user, self.hist = to_device(user), to_device(hist)
        take = slice(None) if shortlist is None else shortlist["take"]
        self.offsets = frame["offsets"] if shortlist is None else shortlist["offsets"]
        self.ids = frame["items"][take]
        self.items = to_device(self.ids)
        self.group = to_device(np.repeat(np.arange(len(rows),dtype=np.int32),np.diff(self.offsets)))
        self.x = to_device(np.clip((frame["x"][take]-scale[0])/scale[1],-8,8))
        self.y = to_device(np.concatenate([r.y for r in rows]).astype(np.float32)) if labeled else None

    def score(self, model, ix):
        g = self.group[ix].long()
        return model(self.user[g],self.items[ix],self.hist[g],self.x[ix])

    @torch.inference_mode()
    def predict(self, model, batch):
        model.eval()
        return torch.cat([self.score(model,slice(i,i+batch)) for i in range(0,len(self.items),batch)]).cpu().numpy()

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ItemEncoder(nn.Module):
    def __init__(self, meta, codes, cfg, content=True):
        super().__init__()
        d = cfg["id_dim"]
        self.content = content
        self.dropout = cfg["id_dropout"]
        self.register_buffer("meta", torch.from_numpy(meta), persistent=False)
        self.register_buffer("codes", torch.from_numpy(codes), persistent=False)
        self.ids = nn.Embedding(int(codes.max()) + 1, d)
        self.cats = nn.ModuleList(nn.Embedding(int(meta[:, i].max())+1, d, padding_idx=0) for i in (0, 1))
        self.text = nn.Embedding(cfg["title_buckets"]+1, d, padding_idx=0)
        self.net = nn.Sequential(nn.Linear(d * (4 if content else 1), 2*d), nn.ReLU(), nn.Linear(2*d, d))

    def forward(self, item):
        codes = self.codes[item.long()].long()
        if self.training and self.dropout:
            codes = codes.masked_fill(torch.rand(codes.shape, device=codes.device) < self.dropout, 0)
        parts = [self.ids(codes)]
        if self.content:
            meta = self.meta[item.long()].long()
            parts.extend(emb(meta[..., j]) for j, emb in enumerate(self.cats))
            tok = meta[..., 2:]
            parts.append(self.text(tok).sum(-2) / (tok != 0).sum(-1, keepdim=True).clamp(min=1))
        return F.normalize(self.net(torch.cat(parts, dim=-1)), dim=-1) * (item != 0).unsqueeze(-1)


class TwoTower(nn.Module):
    def __init__(self, meta, codes, users, cfg, variant):
        super().__init__()
        d = cfg["id_dim"]
        self.with_history = variant != "no_history"
        self.item = ItemEncoder(meta, codes, cfg, content=variant != "no_content")
        self.user = nn.Embedding(users+1, d)
        self.net = nn.Sequential(nn.Linear(d * (2 if self.with_history else 1), 2*d), nn.ReLU(), nn.Linear(2*d, d))

    def query(self, users, histories):
        users = users.long()
        if self.training:
            users = users.masked_fill(torch.rand(users.shape, device=users.device) < .1, 0)
        parts = [self.user(users)]
        if self.with_history:
            h = self.item(histories)
            parts.append(h.sum(1) / (histories != 0).sum(1, keepdim=True).clamp(min=1))
        return F.normalize(self.net(torch.cat(parts, dim=-1)), dim=-1)


def negative_mask(items, rows):
    columns = {}
    for j, item in enumerate(items):
        columns.setdefault(int(item), []).append(j)
    unique = set(columns)
    mask = np.zeros((len(items), len(items)), bool)
    for j, r in enumerate(rows):
        forbidden = unique.intersection(r.seen)
        forbidden.update(unique.intersection(r.positives.tolist()))
        for item in forbidden:
            mask[j, columns[item]] = True
    np.fill_diagonal(mask, False)
    return mask


def contrastive_loss(model, users, histories, items, mask, temperature, extra=None):
    q = model.query(users, histories)
    v = model.item(items)
    scores = ((q @ v.T) / temperature).masked_fill(mask, -1.e9)
    if extra is not None:
        # One verified, exposed nonclick per row, if available.
        neg = (q * model.item(extra)).sum(-1) / temperature
        scores = torch.cat([scores, neg.masked_fill(extra == 0, -1.e9)[:, None]], dim=1)
    return F.cross_entropy(scores, torch.arange(len(items), device=items.device))


@torch.inference_mode()
def item_vectors(model, ids, device):
    model.eval()
    return torch.cat([model.item(torch.as_tensor(np.array(block), device=device))
                      for block in np.array_split(ids, max(1, (len(ids)+16383)//16384))])


@torch.inference_mode()
def user_vectors(model, users, hist, device):
    model.eval()
    out = []
    for lo in range(0, len(users), 1024):
        u = torch.as_tensor(users[lo:lo+1024], device=device)
        h = torch.as_tensor(hist[lo:lo+1024], device=device)
        out.append(model.query(u, h))
    return torch.cat(out)


@torch.inference_mode()
def validation(model, profile, rows, catalog, device, k):
    vectors = item_vectors(model, catalog, device)
    users = user_vectors(model, *profile, device)
    scores, hits, weight = 0., 0, 0
    for lo in range(0, len(rows), 24):
        group = rows[lo:lo+24]
        fetch = min(len(catalog), k + max(len(r.seen) for r in group))
        ix = (users[lo:lo+len(group)] @ vectors.T).topk(fetch).indices.cpu().numpy()
        for r, ids in zip(group, catalog[ix]):
            pos = set(r.positives.tolist())
            if pos:
                retrieved = [i for i in ids.tolist() if i not in r.seen][:k]
                count = len(pos.intersection(retrieved))
                hits += count
                scores += count/len(pos)
                weight += 1
    return scores / max(1, weight), hits

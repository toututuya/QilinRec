"""Full-catalog text retrieval using a residual user tower and a historical popularity prior."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InterestAdapter(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.down = nn.Linear(dim, width)
        self.up = nn.Linear(width, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        x = F.normalize(x.float(), dim=1)
        active = x.ne(0).any(dim=1, keepdim=True)
        return F.normalize(x + self.up(F.gelu(self.down(x))), dim=1) * active


def rank_keys(scores, ids):
    # Higher score wins, then smaller note ID. Sorting float32 bit patterns
    # this way preserves signed numerical order and resolves ties at the cutoff.
    bits = scores.contiguous().view(torch.int32)
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    keys = (ordered.long() << 32) | (0xFFFFFFFF - ids.long())
    return keys.masked_fill(~torch.isfinite(scores), torch.iinfo(torch.int64).min)


class ContentIndex:
    def __init__(self, matrix, ids, device):
        self.ids = np.asarray(ids, dtype=np.int32)
        if len(self.ids) == 0 or np.any(np.diff(self.ids) <= 0):
            raise ValueError("Content index needs nonempty ascending unique IDs")
        self.positions = np.full(len(matrix), -1, np.int32)
        self.positions[self.ids] = np.arange(len(self.ids), dtype=np.int32)
        self.device = device
        self.vectors = torch.as_tensor(matrix[self.ids], device=device, dtype=torch.float16)
        self.item_ids = torch.as_tensor(self.ids, device=device)

    @torch.inference_mode()
    def search(self, queries, rows, k, weights=(0.0,), prior=None):
        # Stream score tiles; every eligible item is scored even for large catalogs.
        if len(rows) != len(queries) or (prior is None and any(weights)):
            raise ValueError("Query alignment or popularity prior missing")
        device = self.device
        vectors = torch.as_tensor(queries, device=device, dtype=torch.float32)
        bias = torch.as_tensor(prior[self.ids], device=device) if prior is not None else None
        output = {
            w: (np.zeros((len(rows), k), np.int32), np.full((len(rows), k), -np.inf, np.float32))
            for w in weights
        }
        low = torch.iinfo(torch.int64).min
        for start in range(0, len(rows), 256):
            group = rows[start : start + 256]
            q = vectors[start : start + len(group)]
            active = q.ne(0).any(dim=1)
            sizes = [len(r.seen) for r in group]
            excluded = np.fromiter((i for r in group for i in r.seen), np.int32, count=sum(sizes))
            loc = self.positions[excluded]
            owner = np.repeat(np.arange(len(group)), sizes)
            best = {w: torch.full((len(group), k), low, dtype=torch.int64, device=device) for w in weights}
            for lo in range(0, len(self.ids), 16384):
                hi = min(lo + 16384, len(self.ids))
                notes = self.vectors[lo:hi].float()
                valid = notes.ne(0).any(dim=1)
                scores = q @ F.normalize(notes, dim=1).T
                blocked = (loc >= lo) & (loc < hi)
                scores[owner[blocked], loc[blocked] - lo] = -torch.inf
                scores.masked_fill_(~active[:, None], -torch.inf)
                for w in weights:
                    adjusted = scores + w * bias[lo:hi] if w else scores.masked_fill(~valid, -torch.inf)
                    keys = rank_keys(adjusted, self.item_ids[lo:hi])
                    top = keys.topk(min(k, keys.shape[1]), sorted=False).values
                    best[w] = torch.cat((best[w], top), dim=1).topk(k).values
            for w, keys in best.items():
                values = keys.cpu().numpy()
                ids = (0xFFFFFFFF - (values & 0xFFFFFFFF)).astype(np.int32)
                ordered = (values >> 32).astype(np.int32)
                bits = (ordered ^ ((ordered >> 31) & 0x7FFFFFFF)).view(np.float32)
                keep = values != low
                output[w][0][start : start + len(group)] = np.where(keep, ids, 0)
                output[w][1][start : start + len(group)] = np.where(keep, bits, -np.inf)
        return output


def combine(semantic, warm, prior, weight):
    # All nonzero-prior items were searched in warm. Any other item outside
    # semantic Top K has K eligible items with at least as high a fused score.
    sid, ss = semantic
    wid, ws = warm
    result = np.zeros_like(sid)
    values = np.full_like(ss, -np.inf)
    for j in range(len(sid)):
        items = np.r_[sid[j], wid[j]]
        scores = np.r_[ss[j] + weight * prior[sid[j]], ws[j]]
        seen = set()
        for n in np.lexsort((items, -scores)):
            item = int(items[n])
            if item > 0 and item not in seen:
                dest = len(seen)
                result[j, dest], values[j, dest] = item, scores[n]
                seen.add(item)
                if len(seen) == result.shape[1]:
                    break
    return result, values

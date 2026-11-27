import math
from collections import defaultdict

import numpy as np
from scipy import sparse


class Popularity:
    def __init__(self, rows, size, cutoff, half_life):
        self.cutoff = cutoff
        self.decay = math.log(2) / (3600 * half_life)
        self.scores = np.zeros(size, np.float32)
        self.first_observed = np.zeros(size, np.int64)
        for r in rows:
            if r.end >= cutoff:
                raise ValueError("Popularity snapshot crosses its cutoff")
            ids = r.items
            first = self.first_observed[ids]
            self.first_observed[ids] = np.where(first == 0, r.end, np.minimum(first, r.end))
            self.scores[r.positives] += math.exp((r.end - cutoff) * self.decay)
        ids = np.flatnonzero(self.scores)
        self.order = ids[np.lexsort((ids, -self.scores[ids]))]

    def multiplier(self, when):
        if when < self.cutoff:
            raise ValueError("Query precedes popularity snapshot")
        return math.exp((self.cutoff - when) * self.decay)

    def retrieve(self, r, k, available):
        factor = self.multiplier(r.start)
        out = []
        for i in self.order:
            if i not in r.seen and available[i]:
                out.append((int(i), float(self.scores[i] * factor)))
                if len(out) == k:
                    break
        return out


def build_itemcf(rows, size, cutoff, cfg):
    half = cfg["cf_half_life_hours"] * 3600
    events = defaultdict(dict)
    for r in rows:
        if r.end >= cutoff:
            raise ValueError("ItemCF snapshot crosses its cutoff")
        user = events[r.uid]
        for i in r.recent:
            user[i] = max(user.get(i, 0), r.start)
        for i in r.positives.tolist():
            user[i] = max(user.get(i, 0), r.end)
    user_row, item_col, values = [], [], []
    for user_no, seen in enumerate(events.values()):
        picked = sorted(seen, key=lambda i: (-seen[i], i))[:cfg["cf_user_cap"]]
        for i in picked:
            user_row.append(user_no)
            item_col.append(i)
            values.append(2 ** ((seen[i] - cutoff) / half))
    matrix = sparse.csr_matrix((np.asarray(values, np.float32), (user_row, item_col)), shape=(len(events), size))
    norm = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel()).clip(1.e-8)
    activity = 1 / np.log2(2 + np.diff(matrix.indptr)).astype(np.float32)
    normalized = matrix.multiply((1 / norm)[None, :]).tocsr()
    weighted = normalized.multiply(activity[:, None]).tocsr()
    left = normalized.T.tocsr()
    active = np.flatnonzero(np.diff(left.indptr))
    indices, weights = [], []
    counts = np.zeros(size, np.int32)
    k = cfg["cf_neighbors"]
    for start in range(0, len(active), 2048):
        block = active[start:start+2048]
        sims = (left[block] @ weighted).tocsr()
        for row, item in enumerate(block):
            lo, hi = sims.indptr[row:row+2]
            ids, scores = sims.indices[lo:hi], sims.data[lo:hi]
            keep = ids != item
            ids, scores = ids[keep], scores[keep]
            order = np.lexsort((ids, -scores))[:k]
            indices.append(ids[order])
            weights.append(scores[order])
            counts[item] = len(order)
    indptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    return sparse.csr_matrix((np.concatenate(weights), np.concatenate(indices), indptr), shape=(size, size))


def cf_scores(graph, hist):
    out = defaultdict(float)
    for rank, i in enumerate(hist):
        lo, hi = graph.indptr[i:i+2]
        weight = 1 / math.sqrt(rank+1)
        for dest, sim in zip(graph.indices[lo:hi].tolist(), graph.data[lo:hi].tolist()):
            out[dest] += sim * weight
    return out


def cf_candidates(scores, seen, available, k):
    filtered = ((i, s) for i, s in scores.items() if i not in seen and available[i])
    return sorted(filtered, key=lambda p: (-p[1], p[0]))[:k]

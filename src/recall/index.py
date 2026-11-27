from time import perf_counter

import faiss
import numpy as np


class CatalogIndex:
    def __init__(self, index, ids):
        self.index = index
        self.ids = ids

    def search(self, vectors, rows, k):
        """Progressively overfetch only for queries that lost neighbors to exclusions."""
        result = [[] for _ in rows]
        pending = np.arange(len(rows))
        fetch = min(k+16, len(self.ids))
        while len(pending):
            d, ix = self.index.search(np.ascontiguousarray(vectors[pending], dtype=np.float32), fetch)
            retry = []
            for j, scores, local in zip(pending, d, ix):
                valid = local >= 0
                ids = self.ids[local[valid]]
                pairs = [(i,s) for i,s in zip(ids.tolist(), scores[valid].tolist()) if i not in rows[j].seen]
                pairs.sort(key=lambda pair: (-pair[1], pair[0]))
                result[j] = pairs[:k]
                if len(pairs) < k and fetch < len(self.ids) and valid.all():
                    retry.append(j)
            pending = np.array(retry, dtype=int)
            fetch = min(2*fetch, len(self.ids))
        return result


def build_indexes(vectors, ids, cfg):
    d = vectors.shape[1]
    start = perf_counter()
    flat = faiss.IndexFlatIP(d)
    for lo in range(0, len(ids), 32768):
        flat.add(np.ascontiguousarray(vectors[ids[lo:lo+32768]]))
    flat_time = perf_counter()-start
    start = perf_counter()
    nlist = min(cfg["ivf_lists"], max(1,len(ids)//40))
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = cfg["seed"]
    index.cp.niter = 12
    rng = np.random.default_rng(cfg["seed"])
    sample = rng.choice(ids, min(len(ids), cfg["ivf_train_max"]), replace=False)
    index.train(np.ascontiguousarray(vectors[sample]))
    for lo in range(0, len(ids), 32768):
        index.add(np.ascontiguousarray(vectors[ids[lo:lo+32768]]))
    index.nprobe = min(nlist,cfg["ivf_probe"])
    return CatalogIndex(flat,ids), CatalogIndex(index,ids), {"flat_build_s":flat_time,"ivf_build_s":perf_counter()-start,
                                                         "nlist":nlist,"nprobe":index.nprobe, "indexed_items":len(ids)}


def batched_search(index, q, rows, k, batch=128):
    result = []
    for lo in range(0, len(rows), batch):
        result.extend(index.search(q[lo:lo+batch], rows[lo:lo+batch], k))
    return result

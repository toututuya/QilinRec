"""FAISS maximum inner-product search for the frozen semantic + popularity score."""

from pathlib import Path

import faiss
import numpy as np


def augmented_articles(matrix, ids, prior, weight):
    vectors = np.ascontiguousarray(matrix[ids], dtype=np.float32)
    faiss.normalize_L2(vectors)
    return np.column_stack((vectors, weight * prior[ids]))


class ContentFaiss:
    def __init__(self, index, ids, popular_ids):
        self.index = index
        self.ids = np.asarray(ids, dtype=np.int32)
        self.popular_ids = np.asarray(popular_ids, dtype=np.int32)
        if index.ntotal != len(self.ids) or np.any(np.diff(self.ids) <= 0):
            raise ValueError("FAISS rows must match unique ascending catalog IDs")

    @classmethod
    def build(cls, kind, matrix, catalog, prior, weight, popular_ids, cfg):
        if kind not in ("flat", "ivf") or weight <= 0:
            raise ValueError("Expected FlatIP or IVFFlat and a positive frozen heat coefficient")
        chunks = np.array_split(catalog, max(1, (len(catalog) + 16383) // 16384))
        ids = np.concatenate([c[matrix[c].any(axis=1) | (prior[c] > 0)] for c in chunks])
        if len(ids) == 0:
            raise ValueError("Empty content catalog")
        dim = matrix.shape[1] + 1
        index = faiss.IndexFlatIP(dim)
        if kind == "ivf":
            nlist = min(cfg["nlist"], max(1, len(ids) // 40))
            index = faiss.IndexIVFFlat(index, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.cp.seed = cfg["seed"]
            index.cp.niter = cfg["iterations"]
            take = np.random.default_rng(cfg["seed"]).choice(
                ids, min(cfg["training_items"], len(ids)), replace=False
            )
            index.train(augmented_articles(matrix, take, prior, weight))
            index.nprobe = min(nlist, cfg["nprobe"])
        for start in range(0, len(ids), 16384):
            index.add(augmented_articles(matrix, ids[start : start + 16384], prior, weight))
        return cls(index, ids, popular_ids[np.isin(popular_ids, ids)])

    def save(self, directory):
        path = Path(directory)
        faiss.write_index(self.index, str(path / "vectors.faiss"))
        np.save(path / "ids.npy", self.ids)
        np.save(path / "popular.npy", self.popular_ids)

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        return cls(
            faiss.read_index(str(path / "vectors.faiss")),
            np.load(path / "ids.npy"),
            np.load(path / "popular.npy"),
        )

    def search(self, queries, rows, k):
        """Return padded internal note IDs. Input queries are the normalized V2 user vectors."""
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.shape != (len(rows), self.index.d - 1) or k < 1 or not np.isfinite(q).all():
            raise ValueError("Invalid queries or candidate budget")
        out = np.zeros((len(q), k), np.int32)
        active = q.any(axis=1)
        for j in np.flatnonzero(~active):
            keep = []
            for item in self.popular_ids:
                if item not in rows[j].seen:
                    keep.append(item)
                    if len(keep) == k:
                        break
            out[j, : len(keep)] = keep
        pending = np.flatnonzero(active)
        q = np.column_stack((q, np.ones(len(q), np.float32)))
        fetch = min(len(self.ids), k + 16)
        while len(pending):
            distances, positions = self.index.search(q[pending], fetch)
            retry = []
            for j, scores, pos in zip(pending, distances, positions, strict=True):
                valid = pos >= 0
                ids, scores = self.ids[pos[valid]], scores[valid]
                keep = np.fromiter((i not in rows[j].seen for i in ids), bool, count=len(ids))
                ids, kept_scores = ids[keep], scores[keep]
                order = np.lexsort((ids, -kept_scores))[:k]
                # Fetch through a boundary tie before applying the global note-ID tiebreak.
                if (
                    valid.all()
                    and fetch < len(self.ids)
                    and (len(order) < k or kept_scores[order[-1]] == scores[-1])
                ):
                    retry.append(j)
                else:
                    out[j, : len(order)] = ids[order]
            pending = np.asarray(retry, dtype=int)
            fetch = min(len(self.ids), 2 * fetch)
        return out

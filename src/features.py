import itertools

import numpy as np

CHANNELS = ("pop", "cf", "tower")
FEATURES = (
    "profile_size_log",
    "history_len",
    "historical_pop",
    "first_seen_age_log_hours",
    "seen_in_fit_exposure",
    "category1_match",
    "category2_match",
    "tower_dot",
    "cf_similarity",
    "pop_rank",
    "cf_rank",
    "tower_rank",
    "pop_returned_score",
    "cf_returned_score",
    "tower_returned_score",
    "pop_source",
    "cf_source",
    "tower_source",
    "rrf",
)


def channel_maps(channels):
    return [{i: (rank, score) for rank, (i, score) in enumerate(row, 1)} for row in channels]


def features(r, ids, maps, cf, pop, meta, vectors, query, offset):
    ids = np.asarray(ids, dtype=np.int32)
    x = np.zeros((len(ids), len(FEATURES)), np.float32)
    x[:, 0] = np.log1p(r.past_count)
    x[:, 1] = len(r.hist)
    x[:, 2] = np.log1p(pop.scores[ids] * pop.multiplier(r.start))
    observed = pop.first_observed[ids]
    valid = observed > 0
    x[valid, 3] = np.log1p((r.start - observed[valid]) / 3600)
    x[:, 4] = valid
    if r.hist:
        cats = meta[np.asarray(r.hist), :2]
        for col in (0, 1):
            query_cats = meta[ids, col]
            x[:, 5 + col] = ((query_cats[:, None] == cats[None, :, col]) & (query_cats[:, None] != 0)).mean(
                axis=1
            )
    x[:, 7] = vectors[ids] @ query
    x[:, 8] = [cf.get(i, 0.0) for i in ids]
    for ch, lookup in enumerate(maps):
        for j, i in enumerate(ids.tolist()):
            value = lookup.get(i)
            if value is not None:
                rank, score = value
                x[j, 9 + ch], x[j, 12 + ch], x[j, 15 + ch] = rank, score, 1
                x[j, 18] += 1 / (offset + rank)
    return x


class FrameWriter:
    def __init__(self):
        self.x, self.items, self.sizes = [], [], []

    def add(self, ids, x):
        self.items.append(ids)
        self.x.append(x)
        self.sizes.append(len(ids))

    def save(self, path):
        np.savez(
            path,
            items=np.concatenate(self.items).astype(np.int32),
            x=np.concatenate(self.x),
            offsets=np.r_[0, np.cumsum(self.sizes)],
            names=np.array(FEATURES),
        )


def rank_groups(scores, items, offsets, k=None):
    groups, selected = [], []
    for a, b in itertools.pairwise(offsets):
        ix = a + np.lexsort((items[a:b], -scores[a:b]))[:k]
        groups.append(items[ix].tolist())
        selected.append(ix)
    return groups, np.concatenate(selected).astype(np.int64)

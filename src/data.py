"""Point-in-time request state and fixed catalog features."""
import hashlib
import heapq
import pickle
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .runtime import STAGES, evenly_spaced, write_json


@dataclass
class Request:
    rid: int
    uid: int
    start: int
    end: int
    items: np.ndarray
    y: np.ndarray
    recent: tuple
    hist: tuple = ()
    seen: frozenset = field(default_factory=frozenset)
    user_code: int = 0

    @property
    def positives(self):
        return self.items[self.y.astype(bool)]


def read_requests(path):
    names = ["request_idx", "user_idx", "recent_clicked_note_idxs", "rec_result_details_with_idx"]
    requests = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=names):
        for r in batch.to_pylist():
            detail = r["rec_result_details_with_idx"]
            pairs = {}
            for row in detail:
                item, y = row["note_idx"] + 1, row["click"]
                if y not in (0, 1):
                    raise ValueError("Missing or invalid click")
                pairs[item] = max(pairs.get(item, 0), y)
            times = [d["request_timestamp"] for d in detail]
            if not all(t is not None and t == int(t) for t in times):
                raise ValueError("Unexpected timestamp precision")
            requests.append(Request(r["request_idx"], r["user_idx"], int(min(times)), int(max(times)),
                                    np.fromiter(pairs, dtype=np.int32), np.fromiter(pairs.values(), dtype=np.uint8),
                                    tuple(i + 1 for i in (r["recent_clicked_note_idxs"] or []))))
    return sorted(requests, key=lambda r: (r.start, r.rid))


def split_data(train, test, cfg):
    cutoff = min(r.start for r in test)
    eligible = [r for r in train if r.end < cutoff]
    cuts = [int(np.quantile([r.start for r in eligible], p)) for p in cfg["split_quantiles"]] + [cutoff]
    data = {}
    lower = -float("inf")
    for name, upper in zip(STAGES, cuts):
        data[name] = [r for r in eligible if lower <= r.start and r.end < upper]
        lower = upper
    data["test"] = test
    summary = dict(cutoffs=cuts, purged_at_test=len(train)-len(eligible),
                   purged_internal=len(eligible)-sum(len(data[k]) for k in STAGES[:-1]))
    if cfg["stage_limits"]:
        for name, cap in zip(STAGES, cfg["stage_limits"], strict=True):
            rows = data[name]
            data[name] = [rows[i] for i in evenly_spaced(len(rows), cap)]
    return data, summary


def attach_histories(data, length):
    # Only these requests may publish profile updates. Their end timestamps
    # upper-bound the time at which a logged request is used as completed.
    sources = data["fit"] + data["rank_train"]
    events = sorted(sources, key=lambda r: (r.end, r.rid))
    profile = defaultdict(dict)
    index = 0
    users = {uid: i + 1 for i, uid in enumerate(sorted({r.uid for r in data["fit"]}))}
    queries = sorted((r for rows in data.values() for r in rows), key=lambda r: (r.start, r.rid))
    for r in queries:
        while index < len(events) and events[index].end < r.start:
            prev = events[index]
            state = profile[prev.uid]
            # A prior snapshot certifies the items were already known at prev.start.
            for item in reversed(prev.recent):
                if item not in state:
                    state[item] = prev.start
            for item in prev.positives.tolist():
                state[item] = max(prev.end, state.get(item, 0))
            index += 1
        state = profile[r.uid]
        r.hist = tuple(heapq.nlargest(length, state, key=lambda i: (state[i], i)))
        # Current official recent list is only a serve-time exclusion list.
        r.seen = frozenset(state).union(r.recent)
        r.user_code = users.get(r.uid, 0)
    return users


def valid_text(s):
    return isinstance(s, str) and s.strip().lower() not in ("", "nan", "none", "null")


def title_ids(title, count, buckets):
    if not valid_text(title):
        return [0] * count
    title = title.strip().lower()
    grams = [title[i:i+2] for i in range(min(len(title), count))]
    return [1 + zlib.crc32(g.encode("utf-8")) % buckets for g in grams] + [0] * (count - len(grams))


def make_catalog(root, cfg):
    paths = sorted(root.glob("notes-train-*-of-00005.parquet"))
    if len(paths) != 5:
        raise ValueError("Expected five catalog shards")
    size = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    # Row 0 is padding. The pinned revision has a dense unique raw ID range from 0.
    meta = np.zeros((size + 1, 2 + cfg["title_tokens"]), np.int32)
    present = np.zeros(size + 1, bool)
    dictionaries = [{}, {}]
    for path in paths:
        cols = ["note_idx", "taxonomy1_id", "taxonomy2_id", "note_title"]
        for b in pq.ParquetFile(path).iter_batches(batch_size=16384, columns=cols):
            ids = b.column(0).to_numpy() + 1
            if ids.min() < 1 or ids.max() > size or present[ids].any() or len(np.unique(ids)) != len(ids):
                raise ValueError("Catalog identifiers differ from the pinned revision")
            present[ids] = True
            for col, mapper in enumerate(dictionaries):
                meta[ids, col] = [mapper.setdefault(s, len(mapper)+1) if valid_text(s) else 0
                                  for s in b.column(col+1).to_pylist()]
            meta[ids, 2:] = [title_ids(t, cfg["title_tokens"], cfg["title_buckets"]) for t in b.column(3).to_pylist()]
    if not present[1:].all():
        raise ValueError("Catalog is incomplete")
    return meta


def prepare(root, run, cfg):
    revision = (root / "revision.txt").read_text(encoding="utf-8-sig").strip()
    if revision != "2d79000cef450b99f6423357ce86e5c132d32dd5":
        raise ValueError("Unexpected Qilin revision")
    train = read_requests(root / "recommendation_train.parquet")
    test = read_requests(root / "recommendation_test.parquet")
    data, summary = split_data(train, test, cfg)
    users = attach_histories(data, cfg["history_length"])
    meta = make_catalog(root, cfg)
    size = len(meta) - 1
    ids = np.arange(1, size+1, dtype=np.int32)
    known = set()
    train_ids = set()
    for r in data["fit"]:
        known.update(r.items.tolist())
        known.update(r.recent)
        train_ids.update(r.positives.tolist())
        train_ids.update(r.hist)
    codes = np.zeros(size + 1, np.int32)
    trained = sorted(train_ids)
    codes[trained] = np.arange(1, len(trained)+1)
    catalog = ids
    if 0 < cfg["catalog_limit"] < size:
        # Smoke catalog uses fit observations and a uniform catalog sample only.
        rng = np.random.default_rng(cfg["seed"])
        observed = np.array(sorted(known), dtype=np.int32)
        cap = cfg["catalog_limit"]
        if len(observed) >= cap:
            catalog = np.sort(rng.choice(observed, cap, replace=False))
        else:
            extra = rng.choice(np.setdiff1d(ids, observed), cap-len(observed), replace=False)
            catalog = np.union1d(observed, extra)
    np.save(run / "meta.npy", meta)
    np.save(run / "codes.npy", codes)
    np.save(run / "catalog.npy", catalog)
    np.save(run / "known.npy", np.array(sorted(known), dtype=np.int32))
    with (run / "requests.pkl").open("wb") as f:
        pickle.dump(data, f, protocol=5)
    summary.update(users=len(users), revision=revision, catalog=len(catalog), catalog_total=size,
                   catalog_previously_observed=len(known), item_codes=len(trained),
                   stages={name: dict(requests=len(rows), exposures=sum(len(r.y) for r in rows),
                                      positives=sum(int(r.y.sum()) for r in rows), with_history=sum(bool(r.hist) for r in rows),
                                      start_min=min(r.start for r in rows), end_max=max(r.end for r in rows),
                                      request_id_sha256=hashlib.sha256(np.array([r.rid for r in rows], dtype="<i8").tobytes()).hexdigest())
                           for name, rows in data.items()})
    write_json(run / "split.json", summary)
    return summary


def load_requests(run):
    with (run / "requests.pkl").open("rb") as f:
        return pickle.load(f)


def profile_arrays(rows, length):
    histories = np.zeros((len(rows), length), np.int32)
    for j, r in enumerate(rows):
        histories[j, :len(r.hist)] = r.hist
    return np.array([r.user_code for r in rows], np.int32), histories

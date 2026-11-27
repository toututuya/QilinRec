"""Audit the official test pool, exact search and fusion using frozen release artifacts."""

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch

from scripts.diagnose_tower import measure, sampled_accuracy
from scripts.train_recall import make_model
from src.data import load_requests
from src.features import FEATURES
from src.metrics import exposure_metrics, retrieval_metrics
from src.prerank import load_frame
from src.recall.classical import Popularity
from src.recall.index import CatalogIndex, batched_search
from src.runtime import write_json


def public_pool(parquet, qrels_path, rows, catalog_size):
    lookup = {r.rid: r for r in rows}
    with qrels_path.open(encoding="utf-8-sig", newline="") as f:
        qrels = defaultdict(set)
        qrels_lines = 0
        for item in csv.DictReader(f):
            qrels[int(item["qid"])].add(int(item["pid"]) + 1)
            qrels_lines += 1
    stored, ids, lens, counts = {}, [], [], Counter()
    for batch in pq.ParquetFile(parquet).iter_batches(
        batch_size=1024, columns=["request_idx", "rec_results"]
    ):
        for rid, candidates in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            r = lookup[rid]
            pool = [i + 1 for i in candidates]
            members = set(pool)
            counts["same_exposure_set"] += members == set(r.items.tolist())
            counts["included_clicks"] += len(members.intersection(r.positives.tolist()))
            counts["qrels_match_requests"] += qrels.get(len(ids), set()) == set(r.positives.tolist())
            counts["duplicate_entries"] += len(pool) - len(members)
            ids.append(rid)
            lens.append(len(pool))
            stored[rid] = pool
    if len(ids) != len(rows) or len(set(ids)) != len(rows) or set(qrels) != set(range(len(rows))):
        raise ValueError("Official query indexing or request membership differs")
    return {
        **counts,
        "requests": len(rows),
        "pool_pairs": sum(lens),
        "pool_min_median_max": np.quantile(lens, [0, 0.5, 1]).tolist(),
        "qrels_raw_rows": qrels_lines,
        "qrels_unique_pairs": sum(map(len, qrels.values())),
        "qrels_sha256": hashlib.sha256(qrels_path.read_bytes()).hexdigest(),
        "stored_order": retrieval_metrics([stored[r.rid] for r in rows], rows, catalog_size, (10, 100)),
    }


def profile_summary(data):
    result = {}
    for key in ("fit", "tower_val", "rank_val", "test"):
        rows = data[key]
        similarities = [
            len(set(r.hist).intersection(r.recent)) / max(1, len(set(r.hist).union(r.recent))) for r in rows
        ]
        result[key] = {
            "requests": len(rows),
            "reconstructed_nonempty": sum(bool(r.hist) for r in rows),
            "official_nonempty": sum(bool(r.recent) for r in rows),
            "mean_jaccard": float(np.mean(similarities)),
            "current_positive_in_official": sum(
                len(set(r.recent).intersection(r.positives.tolist())) for r in rows
            ),
        }
    return result


def compare_fusion(base, stage, rows):
    exp = load_frame(base, stage, "exposures")
    frame = load_frame(base, stage, "candidates")
    labels = np.concatenate([r.y for r in rows]).astype(bool)
    np.testing.assert_array_equal(exp["items"], np.concatenate([r.items for r in rows]))
    channels = [
        exp["x"][labels, FEATURES.index(name)].astype(bool)
        for name in ("pop_source", "cf_source", "tower_source")
    ]
    p, c, t = channels
    fused = []
    for r, (a, b) in zip(rows, itertools.pairwise(frame["offsets"]), strict=True):
        candidates = set(frame["items"][a:b].tolist())
        fused.extend(i in candidates for i in r.positives.tolist())
    f = np.array(fused)
    union = p | c | t
    assert not np.any(f & ~union)
    counts = {
        "positives": int(labels.sum()),
        "popular": int(p.sum()),
        "itemcf": int(c.sum()),
        "tower": int(t.sum()),
        "untruncated_union": int(union.sum()),
        "rrf": int(f.sum()),
        "popular_lost": int((p & ~f).sum()),
        "rrf_new_vs_popular": int((f & ~p).sum()),
        "union_new_vs_popular": int((union & ~p).sum()),
        "mean_popular_items_in_rrf": float(frame["x"][:, FEATURES.index("pop_source")].sum() / len(rows)),
    }
    counts["tower_on_exposures"] = exposure_metrics(
        exp["x"][:, FEATURES.index("tower_dot")], rows, exp["items"], exp["offsets"]
    )
    return counts


def rankers(base):
    out = {}
    for name in ("mlp", "din"):
        info = json.loads((base / f"{name}.json").read_text())
        out[name] = {
            "best_epoch": info["best_epoch"],
            "history": info["history"],
            "test_exposures": info["test"]["exposure"],
            "test_hits10": info["test"]["cascade"]["hits@10"],
        }
    pre = json.loads((base / "prerank.json").read_text())
    out["lambdarank"] = {
        "test_exposures": pre["test"]["lambdarank"]["exposure"],
        "hits50": pre["test"]["lambdarank"]["cascade"]["hits@50"],
        "rrf_hits50": pre["test"]["rrf"]["cascade"]["hits@50"],
    }
    return out


def exact_audit(run, rows):
    base = run / "base"
    cfg = json.loads((run / "config.json").read_text())["config"]
    ids = np.load(run / "catalog.npy")
    vec = np.load(base / "vectors.npy", mmap_mode="r")
    queries = np.load(base / "test.queries.npy")
    faiss.omp_set_num_threads(cfg["faiss_threads"])
    exact = faiss.IndexFlatIP(vec.shape[1])
    norms = []
    for lo in range(0, len(ids), 32768):
        block = np.ascontiguousarray(vec[ids[lo : lo + 32768]])
        norms.extend(np.linalg.norm(block, axis=1))
        exact.add(block)
    common = []
    for index in (exact, faiss.read_index(str(base / "catalog.ivf.index"))):
        found = batched_search(CatalogIndex(index, ids), queries, rows, 100)
        common.append([[i for i, _ in ranking] for ranking in found])
    flat, ivf = common
    credited = [
        (set(a).intersection(r.positives.tolist()), set(b).intersection(r.positives.tolist()))
        for a, b, r in zip(flat, ivf, rows, strict=True)
    ]
    published = json.loads((base / "recall.json").read_text())["test"]["tower"]
    ivf_metrics = retrieval_metrics(ivf, rows, len(ids), (100,))
    if ivf_metrics["hits@100"] != published["hits@100"]:
        raise ValueError("Could not reproduce released approximate search")
    return {
        "flat": retrieval_metrics(flat, rows, len(ids), (100,)),
        "ivf": ivf_metrics,
        "shared_clicks": sum(len(a & b) for a, b in credited),
        "avg_top100_overlap": float(
            np.mean([len(set(a).intersection(b)) / 100 for a, b in zip(flat, ivf, strict=True)])
        ),
        "item_vector_dtype": str(vec.dtype),
        "min_item_norm": float(np.min(norms)),
        "max_item_norm": float(np.max(norms)),
        "all_finite": bool(np.isfinite(norms).all() and np.isfinite(queries).all()),
    }


def original_checkpoints(run, rows, cfg, device):
    catalog = np.load(run / "catalog.npy")
    model = make_model(run, cfg, "base", device)
    out = {}
    for tag in ("best", "last"):
        state = torch.load(run / "base" / f"tower.{tag}.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        out[tag] = {s: measure(model, rows[s], cfg, catalog, device) for s in ("tower_val", "rank_val")}
        out[tag]["fit_inbatch"] = sampled_accuracy(model, rows["fit"], cfg, device, 912)
        out[tag]["validation_inbatch"] = sampled_accuracy(model, rows["tower_val"], cfg, device, 913)
    return out


def protected_popular_replay(run, rows, cfg):
    # Select the size of a guaranteed popular prefix on rank_train. Test labels
    # and the later rank_val window play no role in that choice.
    budgets = (0, 25, 50, 75, 90, 100)
    roster = np.load(run / "catalog.npy")
    size = len(np.load(run / "codes.npy", mmap_mode="r"))
    mask = np.zeros(size, bool)
    mask[roster] = True
    cutoff = json.loads((run / "split.json").read_text())["cutoffs"][0]
    pop = Popularity(rows["fit"], size, cutoff, cfg["pop_half_life_hours"])
    records = {}
    for stage in ("rank_train", "rank_val"):
        with np.load(run / "base" / f"{stage}.candidates.npz") as f:
            ids, offsets = f["items"], f["offsets"]
        hits = Counter()
        macro = Counter()
        shown = {b: set() for b in budgets}
        denom = sum(bool(r.y.any()) for r in rows[stage])
        for r, (a, b) in zip(rows[stage], itertools.pairwise(offsets), strict=True):
            popular = [i for i, _ in pop.retrieve(r, 100, mask)]
            rrf = ids[a:b].tolist()
            positives = set(r.positives.tolist())
            for n in budgets:
                prefix = set(popular[:n])
                picked = popular[:n] + [i for i in rrf if i not in prefix][: 100 - len(prefix)]
                recovered = len(positives.intersection(picked))
                hits[n] += recovered
                if positives:
                    macro[n] += recovered / len(positives)
                shown[n].update(picked)
        records[stage] = {
            n: {"hits100": hits[n], "recall100": macro[n] / denom, "coverage100": len(shown[n]) / len(roster)}
            for n in budgets
        }
        if stage == "rank_train":
            records["selected"] = max(budgets, key=lambda n: records["rank_train"][n]["recall100"])
    return records


def diagnose(args):
    run = args.run
    rows = load_requests(run)
    size = len(np.load(run / "catalog.npy"))
    cfg = json.loads((run / "config.json").read_text())["config"]
    targets = np.concatenate([r.positives for r in rows["test"]])
    codes = np.load(run / "codes.npy", mmap_mode="r")
    out = {
        "pool": public_pool(args.data / "recommendation_test.parquet", args.qrels, rows["test"], size),
        "profiles": profile_summary(rows),
        "exact": exact_audit(run, rows["test"]),
        "fusion": {s: compare_fusion(run / "base", s, rows[s]) for s in ("rank_val", "test")},
        "rankers": rankers(run / "base"),
        "release_checkpoints": original_checkpoints(run, rows, cfg, args.device),
        "fusion_validation_control": protected_popular_replay(run, rows, cfg),
        "test_positive_oov_id": int(np.count_nonzero(codes[targets] == 0)),
    }
    write_json(args.out, out)
    print(out["exact"], out["fusion"]["test"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    diagnose(p.parse_args())

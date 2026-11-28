"""Replay model selection, measure feedback support, and compare recall at matched candidate counts."""

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
from scipy import sparse

from src.data import load_requests
from src.prerank import load_frame
from src.recall.classical import Popularity, cf_candidates, cf_scores
from src.recall.index import CatalogIndex, batched_search
from src.runtime import write_json


def reference_ndcg(rows, predictions):
    discounts = [1 / math.log2(r + 1) for r in range(1, 11)]
    values = []
    for row, scores in zip(rows, predictions, strict=True):
        labels = row.y.tolist()
        positives = sum(labels)
        if positives:
            order = sorted(range(len(labels)), key=lambda j: (-scores[j], row.items[j]))
            dcg = sum(labels[j] * d for j, d in zip(order, discounts))
            values.append(dcg / sum(discounts[:positives]))
    return math.fsum(values) / len(values)


def replay_selection(base, rows, threads):
    state = json.loads((base / "lambdarank.last.json").read_text())
    model = lgb.Booster(model_str=state["model"])
    frame = load_frame(base, "rank_val", "exposures")
    np.testing.assert_array_equal(frame["items"], np.concatenate([r.items for r in rows]))
    splits = frame["offsets"][1:-1]
    curve = []
    for n, logged in enumerate(state["history"], start=1):
        if logged["iteration"] != n:
            raise ValueError("Non-contiguous LambdaRank history")
        scores = model.predict(frame["x"], num_iteration=n, num_threads=threads)
        replayed = reference_ndcg(rows, np.split(scores, splits))
        curve.append({"trees": n, "logged": logged["val_ndcg10"], "reference": replayed})
    best = max(curve, key=lambda step: step["reference"])
    error = max(abs(step["logged"] - step["reference"]) for step in curve)
    if best["trees"] != state["best_iter"] or error > 1e-10:
        raise ValueError(f"Selection mismatch at {best} with maximum difference {error}")
    return {
        "requests": len(rows),
        "exposures": len(frame["items"]),
        "iterations": len(curve),
        "selected": best["trees"],
        "reference_ndcg10": best["reference"],
        "max_abs_diff": error,
        "curve": curve,
    }


def feedback_support(base, stage, rows):
    with np.load(base / f"{stage}.candidates.npz") as frame:
        ids, bounds = frame["items"], frame["offsets"]
    totals = Counter()
    for r, (a, b) in zip(rows, itertools.pairwise(bounds), strict=True):
        generated = set(ids[a:b].tolist())
        labels = [y for i, y in zip(r.items.tolist(), r.y.tolist(), strict=True) if i in generated]
        totals["click"] += sum(labels)
        totals["exposed_nonclick"] += len(labels) - sum(labels)
        totals["unjudged"] += len(generated) - len(labels)
        totals["requests_with_feedback"] += bool(labels)
        totals["requests_with_both_labels"] += 0 < sum(labels) < len(labels)
    return {"requests": len(rows), "pairs": len(ids), **totals}


def catalog_comparison(run, data, cfg):
    rows = data["test"]
    base = run / "base"
    catalog = np.load(run / "catalog.npy")
    codes = np.load(run / "codes.npy", mmap_mode="r")
    available = np.zeros(len(codes), bool)
    available[catalog] = True
    known = set(np.load(run / "known.npy").tolist())
    cutoff = json.loads((run / "split.json").read_text())["cutoffs"][0]
    pop = Popularity(data["fit"], len(codes), cutoff, cfg["pop_half_life_hours"])
    graph = sparse.load_npz(base / "itemcf.npz")
    index = CatalogIndex(faiss.read_index(str(base / "catalog.ivf.index")), catalog)
    k = cfg["recall_k"]
    tower = batched_search(index, np.load(base / "test.queries.npy"), rows, k)
    with np.load(base / "test.candidates.npz") as frame:
        ids, offsets = frame["items"], frame["offsets"]

    slices = {"all": {}, "seen_in_fit_exposures_or_recent": {}, "unseen_in_fit_exposures_and_recent": {}}
    denominators = {name: Counter() for name in slices}
    lengths = []
    for r, tower_list, (a, b) in zip(rows, tower, itertools.pairwise(offsets), strict=True):
        pop_list = [i for i, _ in pop.retrieve(r, 3 * k, available)]
        cf = {i for i, _ in cf_candidates(cf_scores(graph, r.hist), r.seen, available, k)}
        nn = {i for i, _ in tower_list}
        pop_top = set(pop_list[:k])
        merged = pop_top | cf | nn
        lengths.append(len(merged))
        pools = {
            "pop100": pop_top,
            "itemcf100": cf,
            "tower100": nn,
            "rrf100": set(ids[a:b].tolist()),
            "union": merged,
            "pop_matched_to_union_size": set(pop_list[: len(merged)]),
            "pop300": set(pop_list),
        }
        if len(pop_list) != 3 * k or not pools["rrf100"].issubset(merged):
            raise ValueError("Inconsistent catalog replay")
        pos = set(r.positives.tolist())
        for key, relevant in zip(slices, (pos, pos & known, pos - known), strict=True):
            if not relevant:
                continue
            denominators[key].update(positives=len(relevant), click_requests=1)
            for method, candidates in pools.items():
                hits = len(candidates & relevant)
                score = slices[key].setdefault(method, Counter())
                score["hits"] += hits
                score["macro_sum"] += hits / len(relevant)
    for key, methods in slices.items():
        for score in methods.values():
            score["macro_recall"] = score.pop("macro_sum") / denominators[key]["click_requests"]
            score["micro_recall"] = score["hits"] / denominators[key]["positives"]
    return {
        "catalog": len(catalog),
        "input_per_channel": k,
        "mean_union_size": float(np.mean(lengths)),
        "min_union_size": min(lengths),
        "max_union_size": max(lengths),
        "denominators": denominators,
        "slices": slices,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    base = args.run / "base"
    rows = load_requests(args.run)
    cfg = json.loads((args.run / "config.json").read_text())["config"]
    faiss.omp_set_num_threads(cfg["faiss_threads"])
    write_json(
        args.out,
        {
            "lambda_selection": replay_selection(base, rows["rank_val"], cfg["faiss_threads"]),
            "rrf100_feedback": {
                s: feedback_support(base, s, rows[s]) for s in ("rank_train", "rank_val", "test")
            },
            "catalog": catalog_comparison(args.run, rows, cfg),
        },
    )


if __name__ == "__main__":
    main()

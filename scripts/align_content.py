"""Choose how much retrieval order to preserve using two chronological validation subwindows."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from filelock import FileLock

from scripts.verify_run import paired_interval, request_ndcg
from src.data import load_requests
from src.metrics import retrieval_metrics
from src.runtime import write_json

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def order(items, retrieval_position, logits, alpha, offset):
    model_order = np.lexsort((items, -logits))
    rank = np.empty(len(items), dtype=int)
    rank[model_order] = np.arange(1, len(items) + 1)
    recall_order = np.argsort(retrieval_position, kind="stable")
    original = np.empty_like(rank)
    original[recall_order] = np.arange(1, len(items) + 1)
    score = (1 - alpha) / (offset + rank) + alpha / (offset + original)
    return np.lexsort((items, -score))


def replay(raw, indices, alpha, cfg):
    short, top = [], []
    for j in indices:
        a, b = raw["offsets"][j : j + 2]
        ids = raw["items"][a:b]
        pos = np.arange(len(ids))
        chosen = order(ids, pos, raw["prerank"][a:b], alpha, cfg["rrf_offset"])[: cfg["prerank_k"]]
        final = order(ids[chosen], pos[chosen], raw["din"][a:b][chosen], alpha, cfg["rrf_offset"])[
            : cfg["rank_k"]
        ]
        short.append(ids[chosen].tolist())
        top.append(ids[chosen[final]].tolist())
    return short, top


def load_scores(out, stage, rows):
    with np.load(out / f"{stage}.model_scores.npz") as f:
        raw = {k: f[k] for k in f.files}
    np.testing.assert_array_equal(raw["request_ids"], [r.rid for r in rows])
    return raw


def align(args):
    out, run = args.out, args.run
    data = load_requests(run)
    cfg = json.loads((run / "config.json").read_text())["config"]
    size = len(np.load(run / "catalog.npy"))
    val = data["rank_val"]
    raw = load_scores(out, "rank_val", val)
    cutoff = int(np.median([r.start for r in val]))
    fit_ix = [i for i, r in enumerate(val) if r.end < cutoff]
    confirm_ix = [i for i, r in enumerate(val) if r.start >= cutoff]

    def assess(indices, weight):
        group = [val[j] for j in indices]
        _, top = replay(raw, indices, weight, cfg)
        return retrieval_metrics(top, group, size, (10,))

    selection = {str(w): assess(fit_ix, w) for w in GRID}
    proposed = max(GRID, key=lambda w: selection[str(w)]["ndcg@10"])
    confirm = {str(w): assess(confirm_ix, w) for w in dict.fromkeys((0.0, proposed, 1.0))}
    control, trial = confirm["0.0"], confirm[str(proposed)]
    use = (
        proposed if trial["ndcg@10"] > control["ndcg@10"] and trial["hits@10"] >= control["hits@10"] else 0.0
    )
    policy = {
        "grid": GRID,
        "cutoff": cutoff,
        "fit_requests": len(fit_ix),
        "confirmation_requests": len(confirm_ix),
        "purged_crossing_requests": len(val) - len(fit_ix) - len(confirm_ix),
        "selection": selection,
        "proposed_retrieval_weight": proposed,
        "confirmation": confirm,
        "selected_retrieval_weight": use,
        "objective": "macro NDCG@10 on earlier rank_val, then confirm NDCG gain and nondecreasing hits vs frozen ranks",
    }
    write_json(out / "alignment_policy.json", policy)
    result = {"policy": policy}
    frozen = json.loads((out / "cascade.json").read_text())
    for stage in ("rank_val", "test"):
        rows = data[stage]
        scores = raw if stage == "rank_val" else load_scores(out, stage, rows)
        ix = range(len(rows))
        s, t = replay(scores, ix, use, cfg)
        _, base = replay(scores, ix, 0.0, cfg)
        _, direct = replay(scores, ix, 1.0, cfg)
        base_m = retrieval_metrics(base, rows, size, (10,))
        old = frozen[stage]["policies"][frozen["selected"]]["din"]
        np.testing.assert_allclose(
            [base_m["hits@10"], base_m["ndcg@10"]], [old["hits@10"], old["ndcg@10"]], rtol=1e-9, atol=1e-12
        )
        for j, (a, b) in enumerate(itertools.pairwise(scores["offsets"])):
            if not set(t[j]).issubset(s[j]) or not set(s[j]).issubset(scores["items"][a:b]):
                raise ValueError("Aligned results escaped the retrieval pool")
        tmp = out / f"{stage}.aligned.tmp.npz"
        np.savez(
            tmp,
            request_ids=scores["request_ids"],
            short_items=np.concatenate(s).astype(np.int32),
            short_offsets=np.r_[0, np.cumsum(list(map(len, s)))],
            top_items=np.concatenate(t).astype(np.int32),
            top_offsets=np.r_[0, np.cumsum(list(map(len, t)))],
        )
        tmp.replace(out / f"{stage}.aligned.npz")
        result[stage] = {
            "aligned50": retrieval_metrics(s, rows, size, (50,)),
            "aligned10": retrieval_metrics(t, rows, size, (10,)),
            "direct10": retrieval_metrics(direct, rows, size, (10,)),
            "frozen10": base_m,
            "aligned_minus_frozen_ndcg10": paired_interval(
                request_ndcg(rows, t) - request_ndcg(rows, base), rows, cfg["seed"]
            ),
            "aligned_minus_direct_ndcg10": paired_interval(
                request_ndcg(rows, t) - request_ndcg(rows, direct), rows, cfg["seed"]
            ),
        }
    write_json(out / "alignment.json", result)
    print(
        "retrieval weight",
        use,
        "confirmation",
        trial["hits@10"],
        "test",
        result["test"]["aligned10"]["hits@10"],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu", help="Interface compatibility; ranking replay runs on CPU")
    args = p.parse_args()
    with FileLock(args.out / ".content.lock", timeout=0):
        align(args)

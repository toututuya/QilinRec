"""Replay the frozen V2 on Qilin's official recommendation candidates and original qrels."""

import argparse
import ast
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from scripts.align_content import order, replay
from scripts.run_content import profiles, queries
from src.data import load_requests
from src.features import FEATURES
from src.metrics import retrieval_metrics
from src.prerank import load_frame
from src.rank import CTRModel, RankingDataset
from src.recall.classical import Popularity
from src.recall.content import InterestAdapter
from src.runtime import write_json

COMMIT = "299c02667e1dc98dd8e62aa5e13a8e04a9d0363b"
SCORER_SHA = "f5e6c5a65910af4d720c8101b797bcc85177b50f3bc38fd77c191ae1f7eadf77"
QRELS_SHA = "3ffc5bc91420992686836ec5b51cfe725e79652140ef4168b0bcf81433797525"


def author_metric(path):
    # The released evaluator imports training libraries at module scope. Load
    # just its standalone metric from the byte-verified upstream file.
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCORER_SHA:
        raise ValueError("Expected evaluator.py from the pinned Qilin revision")
    fn = next(
        f for f in ast.parse(source).body if isinstance(f, ast.FunctionDef) and f.name == "calculate_metrics"
    )
    globals_ = {"__builtins__": {f.__name__: f for f in (len, max, min, sum, any, enumerate)}}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), globals_)  # noqa: S102 -- SHA-256 pinned
    return globals_["calculate_metrics"]


def official_queries(data, qrels_file, rows):
    if hashlib.sha256(qrels_file.read_bytes()).hexdigest() != QRELS_SHA:
        raise ValueError("Unexpected recommendation qrels")
    qrels = defaultdict(list)
    with qrels_file.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            # Internal identifiers reserve zero for padding; apply the shift to both sides.
            qrels[int(r["qid"])].append(int(r["pid"]) + 1)
    by_id = {r.rid: j for j, r in enumerate(rows)}
    indices, pools = [], []
    parquet = pq.ParquetFile(data / "recommendation_test.parquet")
    for b in parquet.iter_batches(batch_size=1024, columns=["request_idx", "rec_results"]):
        for rid, raw in zip(b.column(0).to_pylist(), b.column(1).to_pylist(), strict=True):
            j = by_id[rid]
            row = rows[j]
            candidates = [i + 1 for i in raw]
            qid = len(indices)  # Original Parquet row, before chronological sorting.
            if (
                len(set(candidates)) != len(candidates)
                or set(candidates) != set(row.items)
                or set(qrels[qid]) != set(row.positives)
            ):
                raise ValueError(f"Pool or relevance mismatch for official query {qid}")
            indices.append(j)
            pools.append(candidates)
    if sorted(indices) != list(range(len(rows))) or set(qrels) != set(range(len(rows))):
        raise ValueError("Incomplete test query coverage")
    return indices, pools, qrels


@torch.inference_mode()
def model_scores(run, content, rows, requests, frame, cfg, device):
    split = json.loads((run / "split.json").read_text())
    policy = json.loads((content / "policy.json").read_text())
    design = json.loads((content / "design.json").read_text())["config"]
    state = torch.load(content / "adapter.best.pt", map_location=device, weights_only=True)
    if state["epoch"] != policy["epoch"]:
        raise ValueError("Selected model mismatch")
    adapter = InterestAdapter(design["dimension"], design["adapter_width"]).to(device)
    adapter.load_state_dict(state["model"])
    q = queries(adapter, profiles(content, "test", rows), device)
    matrix = np.load(content / "articles.npy", mmap_mode="r")
    pop = Popularity(requests["fit"], len(matrix), split["cutoffs"][0], cfg["pop_half_life_hours"])
    ids = frame["items"]
    owners = np.repeat(np.arange(len(rows)), np.diff(frame["offsets"]))
    prior = np.log1p(pop.scores[ids])
    selected = policy["selected"]
    blended = prior.copy()
    if selected != "pop":
        weight = float(selected.removeprefix("w"))
        for a in range(0, len(ids), 8192):
            ix = slice(a, a + 8192)
            articles = torch.as_tensor(matrix[ids[ix]], device=device).float()
            query = torch.as_tensor(q[owners[ix]], device=device)
            sim = (F.normalize(articles, dim=1) * query).sum(1).cpu().numpy()
            score = sim + weight * prior[ix]
            valid = articles.ne(0).any(1).cpu().numpy() | ((weight > 0) & (prior[ix] > 0))
            score[~valid] = -np.inf
            blended[ix] = np.where(q[owners[ix]].any(1), score, prior[ix])
    base = run / "base"
    pre = lgb.Booster(model_file=str(base / "lambdarank.best.txt")).predict(
        frame["x"], num_threads=cfg["faiss_threads"]
    )
    scale = np.load(base / "rank_feature_scale.npy")
    din = CTRModel(
        "din", np.load(base / "vectors.npy"), np.load(run / "codes.npy"), split["users"], cfg, len(scale[0])
    ).to(device)
    din.load_state_dict(torch.load(base / "din.best.pt", map_location=device, weights_only=True)["model"])
    batch = RankingDataset(rows, frame, cfg, scale, device, labeled=False)
    return blended, pre, batch.predict(din, cfg["rank_batch"])


def pool_rankings(frame, blended, pre, din, alpha, cfg):
    result = defaultdict(list)
    ordered_flat = []
    tower_dot = frame["x"][:, FEATURES.index("tower_dot")]
    for a, b in itertools.pairwise(frame["offsets"]):
        ids = frame["items"][a:b]
        initial = np.lexsort((ids, -blended[a:b]))
        ordered_flat.append(initial + a)
        initial_ids = ids[initial]
        positions = np.arange(len(ids))
        coarse = order(initial_ids, positions, pre[a:b][initial], alpha, cfg["rrf_offset"])
        short = coarse[: cfg["prerank_k"]]
        fine = order(initial_ids[short], positions[short], din[a:b][initial][short], alpha, cfg["rrf_offset"])
        # Preserve the native Top 10. Expose a complete order for @100 by
        # appending the coarse tail after the DIN-ordered shortlist.
        ranked = initial_ids[np.r_[short[fine], coarse[len(short) :]]].tolist()
        if len(ranked) != len(ids) or set(ranked) != set(ids):
            raise ValueError("Output must be a permutation of the official candidate pool")
        result["v2_final"].append(ranked)
        result["v2_retrieval_score"].append(initial_ids.tolist())
        result["v1_din"].append(ids[np.lexsort((ids, -din[a:b]))].tolist())
        result["v1_tower_dot"].append(ids[np.lexsort((ids, -tower_dot[a:b]))].tolist())
    flat = np.concatenate(ordered_flat)
    raw = {"items": frame["items"][flat], "offsets": frame["offsets"], "prerank": pre[flat], "din": din[flat]}
    _, native = replay(raw, range(len(result["v2_final"])), alpha, cfg)
    if native != [r[: cfg["rank_k"]] for r in result["v2_final"]]:
        raise ValueError("Full ranking extension changes the native Top 10")
    return result


def evaluate_set(scorer, rankings, chosen, qrels, rows, catalog_size):
    truth = {j: qrels[j] for j in chosen}
    unique = {j: set(v) for j, v in truth.items()}
    out = {}
    for name, preds in rankings.items():
        ks = (10,) if name == "catalog_final_top10" else (10, 100)
        predicted = {j: preds[j] for j in chosen}
        if set(predicted) != set(truth):
            raise ValueError("Missing predictions")
        scores = scorer(predicted, truth, ks)
        clean = scorer(predicted, unique, ks)
        ours = retrieval_metrics([preds[j] for j in chosen], [rows[j] for j in chosen], catalog_size, ks)
        for k in ks:
            for metric in ("MRR", "MAP", "Recall"):
                np.testing.assert_allclose(
                    clean[f"{metric}@{k}"], ours[f"{metric.lower()}@{k}"], rtol=1e-11, atol=1e-12
                )
        out[name] = {"official": scores, "dedup": clean}
    widths = [len(rankings["stored_order"][j]) for j in chosen]
    return {
        "requests": len(truth),
        "exposed_pairs": sum(widths),
        "pool_min_median_max": np.quantile(widths, [0, 0.5, 1]).tolist(),
        "pools_over_50": sum(w > 50 for w in widths),
        "qrels_rows": sum(map(len, truth.values())),
        "unique_clicks": sum(map(len, unique.values())),
        "results": out,
    }


def compare(args):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    cfg = json.loads((args.run / "config.json").read_text())["config"]
    requests = load_requests(args.run)
    rows = requests["test"]
    indices, stored, qrels = official_queries(args.data, args.qrels, rows)
    frame = load_frame(args.run / "base", "test", "exposures")
    np.testing.assert_array_equal(frame["items"], np.concatenate([r.items for r in rows]))
    np.testing.assert_array_equal(np.diff(frame["offsets"]), [len(r.items) for r in rows])
    alpha = json.loads((args.content / "alignment_policy.json").read_text())["selected_retrieval_weight"]
    scores = model_scores(args.run, args.content, rows, requests, frame, cfg, args.device)
    models = pool_rankings(frame, *scores, alpha, cfg)
    request_ids = [r.rid for r in rows]
    selected = json.loads((args.content / "policy.json").read_text())["selected"]
    with np.load(args.content / "test.top100.npz") as f:
        np.testing.assert_array_equal(f["request_ids"], request_ids)
        models["catalog_top100"] = [v[v != 0].tolist() for v in f[selected]]
    with np.load(args.content / "test.aligned.npz") as f:
        np.testing.assert_array_equal(f["request_ids"], request_ids)
        v = f["top_items"]
        models["catalog_final_top10"] = [v[a:b].tolist() for a, b in itertools.pairwise(f["top_offsets"])]
    rankings = {k: [ranks[i] for i in indices] for k, ranks in models.items()}
    rankings["stored_order"] = stored
    ordered = [rows[i] for i in indices]
    size = len(np.load(args.run / "catalog.npy"))
    scorer = author_metric(args.scorer)
    evidence = {
        "upstream_commit": COMMIT,
        "scorer_sha256": SCORER_SHA,
        "qrels_sha256": QRELS_SHA,
        "frozen_policy": selected,
        "frozen_rank_weight": alpha,
        "catalog_size": size,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "torch_version": str(torch.__version__),
    }
    for name, n in (("config_first_1000", 1000), ("all_requests", len(rows))):
        evidence[name] = evaluate_set(scorer, rankings, range(n), qrels, ordered, size)
    full = evidence["all_requests"]["results"]
    orig = json.loads((args.content / "alignment.json").read_text())["test"]["aligned10"]
    din = json.loads((args.run / "base/din.json").read_text())["test"]["exposure"]
    deltas = {}
    # The saved catalog output is deterministic. DIN logits are recomputed and
    # close scores may change order across devices. Bound aggregate drift by
    # one query's maximum contribution and retain the measured differences.
    for key, old, atol in (("v1_din", din, 1 / len(rows)), ("catalog_final_top10", orig, 1e-12)):
        deltas[key] = {}
        for metric in ("MRR", "MAP"):
            actual, previous = full[key]["dedup"][f"{metric}@10"], old[f"{metric.lower()}@10"]
            np.testing.assert_allclose(actual, previous, rtol=0, atol=atol)
            deltas[key][f"{metric}@10"] = actual - previous
    evidence["dedup_minus_previous_release"] = deltas
    write_json(args.out, evidence)
    for name in ("config_first_1000", "all_requests"):
        print(name, evidence[name]["results"]["v2_final"]["official"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "content", "data", "qrels", "scorer", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    compare(p.parse_args())

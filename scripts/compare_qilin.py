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
from scripts.run_content import profiles, queries, report, stage_pools
from scripts.verify_run import paired_interval, request_ndcg
from src.data import load_requests
from src.features import FEATURES
from src.metrics import retrieval_metrics
from src.prerank import load_frame
from src.rank import CTRModel, RankingDataset
from src.recall.classical import Popularity
from src.recall.content import ContentIndex, InterestAdapter
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


class FrozenBGE(torch.nn.Module):
    def forward(self, encoded):
        return F.normalize(encoded.float(), dim=1)


def content_models(content, policy):
    design = json.loads((content / "design.json").read_text())["config"]
    state = torch.load(content / "adapter.best.pt", map_location="cpu", weights_only=True)
    if state["epoch"] != policy["epoch"]:
        raise ValueError("Selected model mismatch")
    adapter = InterestAdapter(design["dimension"], design["adapter_width"])
    adapter.load_state_dict(state["model"])
    return {"bge": FrozenBGE(), "bge_adapter": adapter}


@torch.inference_mode()
def exposure_content_scores(frame, user_vectors, matrix, prior, weight, device):
    """Score both user encoders against exactly the same logged notes."""
    ids = frame["items"]
    owners = np.repeat(np.arange(len(frame["offsets"]) - 1), np.diff(frame["offsets"]))
    scores = {key + suffix: np.empty(len(ids), np.float32) for key in user_vectors for suffix in ("", "_pop")}
    for a in range(0, len(ids), 8192):
        ix = slice(a, a + 8192)
        articles = F.normalize(torch.as_tensor(matrix[ids[ix]], device=device).float(), dim=1)
        has_text = articles.ne(0).any(1).cpu().numpy()
        heat = prior[ids[ix]]
        for name, vectors in user_vectors.items():
            batch = vectors[owners[ix]]
            active = batch.any(axis=1)
            similarity = (articles * torch.as_tensor(batch, device=device)).sum(1).cpu().numpy()
            scores[name][ix] = np.where(active & has_text, similarity, -np.inf)
            blended = np.where(has_text | (heat > 0), similarity + weight * heat, -np.inf)
            scores[name + "_pop"][ix] = np.where(active, blended, heat)
    return scores


def ranker_scores(run, rows, frame, cfg, users, device):
    base = run / "base"
    pre = lgb.Booster(model_file=str(base / "lambdarank.best.txt")).predict(
        frame["x"], num_threads=cfg["faiss_threads"]
    )
    scale = np.load(base / "rank_feature_scale.npy")
    din = CTRModel(
        "din", np.load(base / "vectors.npy"), np.load(run / "codes.npy"), users, cfg, len(scale[0])
    ).to(device)
    din.load_state_dict(torch.load(base / "din.best.pt", map_location=device, weights_only=True)["model"])
    batch = RankingDataset(rows, frame, cfg, scale, device, labeled=False)
    return pre, batch.predict(din, cfg["rank_batch"])


def pool_rankings(frame, blended, ablations, pre, din, alpha, cfg):
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
        for name, scores in ablations.items():
            result[name].append(ids[np.lexsort((ids, -scores[a:b]))].tolist())
    flat = np.concatenate(ordered_flat)
    raw = {"items": frame["items"][flat], "offsets": frame["offsets"], "prerank": pre[flat], "din": din[flat]}
    _, native = replay(raw, range(len(result["v2_final"])), alpha, cfg)
    if native != [r[: cfg["rank_k"]] for r in result["v2_final"]]:
        raise ValueError("Full ranking extension changes the native Top 10")
    return result


def evaluate_set(scorer, rankings, chosen, qrels, rows, catalog_size, seed):
    truth = {j: qrels[j] for j in chosen}
    unique = {j: set(v) for j, v in truth.items()}
    out = {}
    group = [rows[j] for j in chosen]
    ndcgs = {}
    for name, preds in rankings.items():
        ks = (10,) if name == "catalog_final_top10" else (10, 100)
        if name == "catalog_top100":
            ks = (10, 50, 100)
        predicted = {j: preds[j] for j in chosen}
        if set(predicted) != set(truth):
            raise ValueError("Missing predictions")
        scores = scorer(predicted, truth, ks)
        clean = scorer(predicted, unique, ks)
        predictions = list(predicted.values())
        ours = retrieval_metrics(predictions, group, catalog_size, ks)
        ndcgs[name] = request_ndcg(group, predictions)
        np.testing.assert_allclose(ndcgs[name].mean(), ours["ndcg@10"], rtol=1e-11, atol=1e-12)
        for k in ks:
            for metric in ("MRR", "MAP", "Recall"):
                np.testing.assert_allclose(
                    clean[f"{metric}@{k}"], ours[f"{metric.lower()}@{k}"], rtol=1e-11, atol=1e-12
                )
        out[name] = {"official": scores, "dedup": clean, "NDCG@10": ours["ndcg@10"]}
    widths = [len(rankings["stored_order"][j]) for j in chosen]
    return {
        "requests": len(truth),
        "exposed_pairs": sum(widths),
        "pool_min_median_max": np.quantile(widths, [0, 0.5, 1]).tolist(),
        "pools_over_50": sum(w > 50 for w in widths),
        "qrels_rows": sum(map(len, truth.values())),
        "unique_clicks": sum(map(len, unique.values())),
        "results": out,
        "paired_ndcg10": {
            f"{trial}_minus_{control}": paired_interval(ndcgs[trial] - ndcgs[control], group, seed)
            for trial, control in (
                ("bge_adapter", "bge"),
                ("bge_pop", "bge"),
                ("bge_adapter_pop", "bge_adapter"),
                ("bge_adapter_pop", "bge_pop"),
            )
        },
    }


def catalog_ablations(content, device, rows, models, matrix, catalog, pop, prior, weight, saved):
    """Use the production catalog search, eligibility checks and historical popularity fallback."""
    args = argparse.Namespace(out=content, device=device)
    available = np.zeros(len(matrix), bool)
    available[catalog] = True
    index = ContentIndex(matrix, catalog, device)
    warm = ContentIndex(matrix, catalog[prior[catalog] > 0], device)
    k = saved["pop"].shape[1]
    fused_key = f"w{weight:g}"
    metrics, replay = {}, {}
    for name, model in models.items():
        pools = stage_pools(
            args, {"k": k}, "test", rows, index, warm, model.to(device), pop, available, prior, [weight]
        )
        np.testing.assert_array_equal(pools["pop"], saved["pop"])
        for key, output in (("semantic", name), (fused_key, name + "_pop")):
            metrics[output] = report(pools[key], rows, len(catalog))
        if name == "bge_adapter" and fused_key in saved:
            replay["identical_ordered_top100_requests"] = int(
                (pools[fused_key] == saved[fused_key]).all(axis=1).sum()
            )
            replay["requests"] = len(rows)
    metrics["pop"] = report(saved["pop"], rows, len(catalog))
    return {"metrics": metrics, "deployed_replay": replay}


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
    split = json.loads((args.run / "split.json").read_text())
    policy = json.loads((args.content / "policy.json").read_text())
    selected = policy["selected"]
    selected_weight = 0.0 if selected == "pop" else float(selected.removeprefix("w"))
    # The released model selected 0.03 on pre-test windows. Keep the same
    # predeclared ablation if another run selected pure heat or zero weight.
    weight = selected_weight or 0.03
    matrix = np.load(args.content / "articles.npy", mmap_mode="r")
    pop = Popularity(requests["fit"], len(matrix), split["cutoffs"][0], cfg["pop_half_life_hours"])
    prior = np.log1p(pop.scores)
    encoders = content_models(args.content, policy)
    encoded = profiles(args.content, "test", rows)
    vectors = {name: queries(model.to(args.device), encoded, args.device) for name, model in encoders.items()}
    has_profile = encoded.any(axis=1)
    for v in vectors.values():
        np.testing.assert_array_equal(v.any(axis=1), has_profile)
    ablations = exposure_content_scores(frame, vectors, matrix, prior, weight, args.device)
    deployed = ablations["bge_adapter_pop"]
    if selected == "pop":
        deployed = prior[frame["items"]]
    elif selected_weight == 0:
        active = np.repeat(has_profile, np.diff(frame["offsets"]))
        deployed = np.where(active, ablations["bge_adapter"], prior[frame["items"]])
    alpha = json.loads((args.content / "alignment_policy.json").read_text())["selected_retrieval_weight"]
    pre, din = ranker_scores(args.run, rows, frame, cfg, split["users"], args.device)
    models = pool_rankings(frame, deployed, ablations, pre, din, alpha, cfg)
    request_ids = [r.rid for r in rows]
    with np.load(args.content / "test.top100.npz") as f:
        np.testing.assert_array_equal(f["request_ids"], request_ids)
        saved = {key: f[key] for key in {"pop", selected}}
        models["catalog_top100"] = [v[v != 0].tolist() for v in saved[selected]]
    with np.load(args.content / "test.aligned.npz") as f:
        np.testing.assert_array_equal(f["request_ids"], request_ids)
        v = f["top_items"]
        models["catalog_final_top10"] = [v[a:b].tolist() for a, b in itertools.pairwise(f["top_offsets"])]
    rankings = {k: [ranks[i] for i in indices] for k, ranks in models.items()}
    rankings["stored_order"] = stored
    ordered = [rows[i] for i in indices]
    catalog = np.load(args.run / "catalog.npy")
    catalog_device = args.catalog_device or args.device
    scorer = author_metric(args.scorer)
    evidence = {
        "upstream_commit": COMMIT,
        "scorer_sha256": SCORER_SHA,
        "qrels_sha256": QRELS_SHA,
        "frozen_policy": selected,
        "frozen_rank_weight": alpha,
        "adapter_epoch": policy["epoch"],
        "ablation_pop_weight": weight,
        "catalog_size": len(catalog),
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "catalog_device": torch.cuda.get_device_name(catalog_device)
        if catalog_device.startswith("cuda")
        else catalog_device,
        "torch_version": str(torch.__version__),
    }
    for name, n in (("config_first_1000", 1000), ("all_requests", len(rows))):
        evidence[name] = evaluate_set(scorer, rankings, range(n), qrels, ordered, len(catalog), cfg["seed"])
        evidence[name]["empty_text_profiles"] = int((~has_profile[indices[:n]]).sum())
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
    evidence["full_catalog_ablation"] = catalog_ablations(
        args.content, catalog_device, rows, encoders, matrix, catalog, pop, prior, weight, saved
    )
    write_json(args.out, evidence)
    for name in ("config_first_1000", "all_requests"):
        print(name, evidence[name]["results"]["v2_final"]["official"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "content", "data", "qrels", "scorer", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--catalog-device", help="Device for full-catalog search (defaults to --device)")
    compare(p.parse_args())

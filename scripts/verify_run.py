"""Replay each saved test shortlist and verify its membership and hit denominators."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import load_requests
from src.features import FEATURES, rank_groups
from src.metrics import exposure_metrics, retrieval_metrics
from src.prerank import load_frame
from src.rank import CTRModel, RankingDataset
from src.runtime import write_json


def read(path):
    return json.loads(path.read_text())


def request_ndcg(rows, groups, k=10):
    discount = 1 / np.log2(np.arange(2, k + 2))
    out = []
    for r, top in zip(rows, groups, strict=True):
        relevant = set(r.positives.tolist())
        gain = sum(discount[j] for j, i in enumerate(top[:k]) if i in relevant)
        out.append(gain / discount[: len(relevant)].sum() if relevant else 0.0)
    return np.array(out)


def paired_interval(delta, rows, seed, samples=2000):
    # Resample users, preserving all their requests; estimate request-macro NDCG.
    valid = np.array([bool(r.y.any()) for r in rows])
    _, groups = np.unique([r.uid for r in rows], return_inverse=True)
    sums = np.bincount(groups[valid], weights=delta[valid], minlength=groups.max() + 1)
    counts = np.bincount(groups[valid], minlength=len(sums))
    rng = np.random.default_rng(seed)
    distribution = []
    for _ in range(samples // 50):
        draws = rng.integers(len(sums), size=(50, len(sums)))
        distribution.extend(sums[draws].sum(1) / counts[draws].sum(1))
    return {
        "delta_ndcg10": float(delta[valid].mean()),
        "cluster": "user",
        "users": len(sums),
        "replicates": samples,
        "confidence95": np.quantile(distribution, [0.025, 0.975]).tolist(),
    }


def replay_ctrs(run, rows, cfg, device):
    base = run / "base"
    frame = load_frame(base, "test", "exposures")
    data = RankingDataset(rows, frame, cfg, np.load(base / "rank_feature_scale.npy"), device)
    vectors, codes = np.load(base / "vectors.npy"), np.load(run / "codes.npy")
    users = read(run / "split.json")["users"]
    outcomes = {}
    for name in ("mlp", "din"):
        model = CTRModel(name, vectors, codes, users, cfg, len(FEATURES)).to(device)
        state = torch.load(base / f"{name}.best.pt", weights_only=True, map_location=device)
        model.load_state_dict(state["model"])
        logits = data.predict(model, cfg["rank_batch"])
        metrics = exposure_metrics(logits, rows, data.ids, data.offsets, logits=True)
        logged = read(base / f"{name}.json")["test"]["exposure"]
        for key in ("auc", "gauc", "logloss", "ndcg@10", "mrr@10", "map@10"):
            np.testing.assert_allclose(metrics[key], logged[key], rtol=2e-6, atol=1e-7)
        groups, _ = rank_groups(logits, data.ids, data.offsets, 10)
        outcomes[name] = request_ndcg(rows, groups)
    return paired_interval(outcomes["din"] - outcomes["mlp"], rows, cfg["seed"])


def verify(run, device):
    base = run / "base"
    requests = load_requests(run)
    rows = requests["test"]
    c = read(run / "config.json")["config"]
    catalog = np.load(run / "catalog.npy")
    available = set(catalog.tolist())
    report = {}
    ndcg = {}
    with np.load(base / "test.candidates.npz") as logged:
        if list(logged["names"]) != list(FEATURES):
            raise AssertionError("Feature schema mismatch")
        vals = {n: logged[n] for n in ["items", "x", "offsets"]}
    last = None
    stages = [
        ("rrf", vals, read(base / "recall.json")["test"]["rrf"], c["recall_k"]),
        (
            "lambdarank",
            base / "test.shortlist.npz",
            read(base / "prerank.json")["test"]["lambdarank"]["cascade"],
            c["prerank_k"],
        ),
        ("mlp", base / "test.mlp.top.npz", read(base / "mlp.json")["test"]["cascade"], c["rank_k"]),
        ("din", base / "test.din.top.npz", read(base / "din.json")["test"]["cascade"], c["rank_k"]),
    ]
    for method, source, reference, k in stages:
        if isinstance(source, dict):
            f = source
        else:
            with np.load(source) as z:
                f = {n: z[n] for n in z.files}
            np.testing.assert_array_equal(f["request_ids"], [r.rid for r in rows])
        offsets, items = f["offsets"], f["items"]
        assert len(offsets) == len(rows) + 1 and offsets[0] == 0 and offsets[-1] == len(items)
        groups = []
        for j, (r, a, b) in enumerate(zip(rows, offsets[:-1], offsets[1:], strict=True)):
            ids = items[a:b].tolist()
            members = set(ids)
            assert len(ids) <= k and len(members) == len(ids)
            assert available.issuperset(members) and not members.intersection(r.seen)
            if last is not None:
                assert last[j].issuperset(members)
            groups.append(ids)
        if method in ("rrf", "lambdarank"):
            last = [set(g) for g in groups]
        if method == "rrf":
            ranks = f["x"][:, 9:12]
            np.testing.assert_array_equal(ranks > 0, f["x"][:, 15:18].astype(bool))
            np.testing.assert_allclose(
                f["x"][:, -1], np.where(ranks > 0, 1 / (c["rrf_offset"] + ranks), 0.0).sum(1), atol=1e-7
            )
        actual = retrieval_metrics(groups, rows, len(catalog), (k,))
        if method in ("mlp", "din"):
            ndcg[method] = request_ndcg(rows, groups)
        for name in ["positives", "requests", f"recall@{k}", f"ndcg@{k}", f"hits@{k}", f"retained@{k}"]:
            np.testing.assert_allclose(reference[name], actual[name], rtol=1e-9)
        report[method] = {
            "k": k,
            "requests": len(rows),
            "positives": actual["positives"],
            "hits": actual[f"hits@{k}"],
            "min_returned": int(np.diff(offsets).min()),
            "max_returned": int(np.diff(offsets).max()),
        }
    fit_clicked = {i for r in requests["fit"] for i in r.positives.tolist()}
    known = set(np.load(run / "known.npy").tolist())
    counts = {"outside_catalog": 0, "excluded_seen": 0, "not_in_fit_observations": 0, "not_in_fit_clicks": 0}
    for r in rows:
        pos = set(r.positives.tolist())
        counts["outside_catalog"] += len(pos - available)
        counts["excluded_seen"] += len(r.seen.intersection(pos))
        counts["not_in_fit_observations"] += len(pos - known)
        counts["not_in_fit_clicks"] += len(pos - fit_clicked)
    report["positive_diagnostics"] = counts
    report["din_minus_mlp"] = {
        "exposure": replay_ctrs(run, rows, c, device),
        "cascade": paired_interval(ndcg["din"] - ndcg["mlp"], rows, c["seed"]),
    }
    write_json(run / "verification.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    verify(args.run, args.device)

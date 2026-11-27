"""Verify saved content-cascade results and export aggregate evidence and experiment rows."""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from scripts.verify_run import paired_interval, request_ndcg
from src.data import load_requests
from src.metrics import retrieval_metrics
from src.runtime import write_json


def read(path):
    return json.loads(path.read_text())


def groups(flat, bounds):
    return [flat[a:b].tolist() for a, b in itertools.pairwise(bounds)]


def verify(run, out, cascade, alignment):
    requests = load_requests(run)
    catalog = np.load(run / "catalog.npy")
    allowed = set(catalog.tolist())
    checked = {}
    for stage in ("rank_val", "test"):
        rows = requests[stage]
        checked[stage] = {"requests": len(rows), "positives": sum(int(r.y.sum()) for r in rows)}
        with np.load(out / f"{stage}.top100.npz") as pool:
            np.testing.assert_array_equal(pool["request_ids"], [r.rid for r in rows])
            for method, reference in cascade[stage]["policies"].items():
                parent = [[int(i) for i in g if i] for g in pool[method]]
                with np.load(out / f"{stage}.{method}.cascade.npz") as f:
                    np.testing.assert_array_equal(f["request_ids"], pool["request_ids"])
                    short = groups(f["shortlist_items"], f["shortlist_offsets"])
                    ranking = {
                        "retrieval": parent,
                        "lambdarank": short,
                        **{n: groups(f[n + "_items"], f[n + "_offsets"]) for n in ("mlp", "din")},
                    }
                for j, r in enumerate(rows):
                    sets = {n: set(g[j]) for n, g in ranking.items()}
                    if (
                        not sets["retrieval"].issubset(allowed)
                        or sets["retrieval"].intersection(r.seen)
                        or not sets["lambdarank"].issubset(sets["retrieval"])
                        or not (sets["mlp"] | sets["din"]).issubset(sets["lambdarank"])
                    ):
                        raise ValueError("Candidate membership violation")
                    if any(len(sets[n]) != len(g[j]) for n, g in ranking.items()):
                        raise ValueError("Duplicate item within request")
                for n, k in (("retrieval", 100), ("lambdarank", 50), ("mlp", 10), ("din", 10)):
                    if max(map(len, ranking[n])) > k:
                        raise ValueError("Candidate budget exceeded")
                    actual = retrieval_metrics(ranking[n], rows, len(catalog), (k,))
                    for key in actual:
                        np.testing.assert_allclose(actual[key], reference[n][key], rtol=1e-10, atol=1e-12)
            with np.load(out / f"{stage}.aligned.npz") as f:
                np.testing.assert_array_equal(f["request_ids"], pool["request_ids"])
                short = groups(f["short_items"], f["short_offsets"])
                top = groups(f["top_items"], f["top_offsets"])
            for source, s, t in zip(pool[cascade["selected"]], short, top, strict=True):
                if (
                    not set(t).issubset(s)
                    or not set(s).issubset(source)
                    or len(t) != len(set(t))
                    or len(s) != len(set(s))
                    or len(t) > 10
                    or len(s) > 50
                ):
                    raise ValueError("Invalid aligned cascade")
            for name, k, ranking in (("aligned50", 50, short), ("aligned10", 10, top)):
                for key, value in retrieval_metrics(ranking, rows, len(catalog), (k,)).items():
                    np.testing.assert_allclose(value, alignment[stage][name][key], rtol=1e-10, atol=1e-12)
            with np.load(run / "base" / f"{stage}.din.top.npz") as legacy:
                np.testing.assert_array_equal(legacy["request_ids"], pool["request_ids"])
                v1 = groups(legacy["items"], legacy["offsets"])
            checked[stage]["aligned_minus_v1_din"] = paired_interval(
                request_ndcg(rows, top) - request_ndcg(rows, v1), rows, 20261127
            )
    return checked


def export(args):
    out, run, dest = args.out, args.run, args.dest
    evidence = {
        n: read(out / f"{n}.json")
        for n in ("design", "text_ready", "training", "proposed", "policy", "test", "cascade", "alignment")
    }
    cascade = evidence["cascade"]
    evidence["verified"] = verify(run, out, cascade, evidence["alignment"])
    evidence["dependencies"] = {
        n: importlib.metadata.version(n)
        for n in (
            "torch",
            "transformers",
            "tokenizers",
            "safetensors",
            "numpy",
            "faiss-cpu",
            "lightgbm",
            "pyarrow",
        )
    }
    evidence["replay_device"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    artifacts = [
        "articles.npy",
        "adapter.best.pt",
        "adapter.last.pt",
        "rank_val.top100.npz",
        "test.top100.npz",
    ]
    artifacts += sorted(p.name for p in out.glob("*.cascade.npz"))
    artifacts += sorted(p.name for p in out.glob("*.aligned.npz"))
    evidence["sha256"] = {}
    for file in artifacts:
        with (out / file).open("rb") as f:
            evidence["sha256"][file] = hashlib.file_digest(f, "sha256").hexdigest()
    write_json(dest / "evidence.json", evidence)
    table = []

    def add(stage, method, layer, m, k):
        table.append(
            {
                "window": stage,
                "method": method,
                "stage": layer,
                "k": k,
                "requests": m["requests"],
                "observed_clicks": m["positives"],
                "hits": m[f"hits@{k}"],
                **{n: m[f"{n}@{k}"] for n in ("recall", "retained", "hr", "ndcg", "coverage")},
            }
        )

    base = run / "base"
    windows = {
        "rank_train": evidence["proposed"]["rank_train"],
        "rank_val": evidence["policy"]["rank_val"],
        "test": evidence["test"]["metrics"],
    }
    for stage in ("rank_train", "rank_val", "test"):
        for n, m in windows[stage].items():
            add(stage, n, "retrieval", m, 100)
        add(stage, "v1_rrf", "retrieval", read(base / "recall.json")[stage]["rrf"], 100)
        if stage != "rank_train":
            for n, parts in cascade[stage]["policies"].items():
                for s, k in (("lambdarank", 50), ("mlp", 10), ("din", 10)):
                    add(stage, n, s, parts[s], k)
            add(
                stage, "v1_rrf", "lambdarank", read(base / "prerank.json")[stage]["lambdarank"]["cascade"], 50
            )
            for n in ("mlp", "din"):
                add(stage, "v1_rrf", n, read(base / f"{n}.json")[stage]["cascade"], 10)
            for n, k in (("aligned50", 50), ("aligned10", 10), ("direct10", 10)):
                add(stage, cascade["selected"], n, evidence["alignment"][stage][n], k)
    with (dest / "experiments.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print("Verified", evidence["verified"], len(table), "experiment rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "out", "dest"):
        parser.add_argument("--" + key, type=Path, required=True)
    export(parser.parse_args())

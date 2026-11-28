"""Verify the bounded final experiments against release artifacts and export aggregate evidence."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.align_content import load_scores, replay
from scripts.content_ann import overlap
from scripts.run_content import report
from scripts.verify_run import paired_interval
from src.data import load_requests
from src.runtime import write_json


def read(path):
    return json.loads(path.read_text())


def quality(run, content, ann, rows, selected, catalog_size):
    archive = read(content / "test.json")
    index_ids = np.load(ann / "flat/ids.npy")
    np.testing.assert_array_equal(index_ids, np.load(ann / "ivf/ids.npy"))
    with np.load(content / "test.top100.npz") as source:
        np.testing.assert_array_equal(source["request_ids"], [r.rid for r in rows])
        exact = source[selected]
    result = {}
    for name in ("flat", "ivf"):
        info = read(ann / f"{name}.test.json")
        with np.load(ann / name / "test.top100.npz") as source:
            np.testing.assert_array_equal(source["request_ids"], [r.rid for r in rows])
            ids = source[selected]
        measured = report(ids, rows, catalog_size)
        for key, number in measured.items():
            np.testing.assert_allclose(number, info["metrics"][key], rtol=0, atol=1e-12)
        assert overlap(ids, exact)["mean_top100_overlap"] == info["mean_top100_overlap"]
        assert info["indexed_items"] == len(index_ids)
        if name == "flat":
            assert measured["hits@100"] == archive["metrics"][selected]["hits@100"]
        delta = []
        for r, actual, reference in zip(rows, ids, exact, strict=True):
            clicked = set(r.positives.tolist())
            delta.append(
                (len(clicked.intersection(actual.tolist())) - len(clicked.intersection(reference.tolist())))
                / max(1, len(clicked))
            )
        interval = paired_interval(np.asarray(delta), rows, info["inputs"]["config"]["seed"])
        interval["delta_macro_recall100"] = interval.pop("delta_ndcg10")
        result[name] = {**info, "minus_exact_recall100": interval}
    latency = {kind: read(ann / f"{kind}.latency.json") for kind in ("torch", "flat", "ivf")}
    for field in ("request_sha256", "requests", "warmup", "threads", "batch_size", "cpu_affinity"):
        assert latency["torch"][field] == latency["flat"][field] == latency["ivf"][field]
    for kind, bench in latency.items():
        assert 0 < bench["p50_ms"] <= bench["p95_ms"]
        if kind != "torch":
            artifacts = ["vectors.faiss", "ids.npy", "popular.npy"]
            assert sum((ann / kind / name).stat().st_size for name in artifacts) == bench["index_files_bytes"]
    cascade = read(ann / "ivf.cascade.json")
    alpha = read(content / "alignment_policy.json")["selected_retrieval_weight"]
    assert cascade["frozen_retrieval_rank_weight"] == alpha
    cfg = read(run / "config.json")["config"]
    raw = load_scores(ann / "ivf", "test", rows)
    _, top = replay(raw, range(len(rows)), alpha, cfg)
    with np.load(ann / "ivf/test.aligned.npz") as saved:
        np.testing.assert_array_equal(saved["top_items"], np.concatenate(top))
        np.testing.assert_array_equal(saved["top_offsets"], np.r_[0, np.cumsum(list(map(len, top)))])
    final = report(top, rows, catalog_size, (10,))
    np.testing.assert_allclose(final["ndcg@10"], cascade["test"]["final"]["ndcg@10"], rtol=0, atol=1e-12)
    return result, latency, cascade


def run(args):
    study, ann = args.study, args.study / "content-ann"
    split = read(args.run / "split.json")
    rows = load_requests(args.run)["test"]
    controls = read(args.controls)
    feedback = read(study / "feedback.json")
    text = read(study / "text_complete.json")
    selected = read(args.content / "policy.json")["selected"]
    retrieval, latency, cascade = quality(
        args.run, args.content, ann, rows, selected, controls["catalog_size"]
    )
    archived = read(args.content / "test.json")
    assert archived["slices"][selected] == feedback["test"]["fit_visibility"]
    for stage, counts in feedback.items():
        assert counts["requests"] == split["stages"][stage]["requests"]
        assert counts["clicked"] + counts["exposed_nonclick"] + counts["unknown"] == counts["candidates"]
        assert sum(g["hits"] for g in counts["fit_visibility"].values()) == counts["clicked"]
    table = []

    def add(task, cohort, method, **fields):
        table.append(
            {"task": task, "cohort": cohort, "method": method, **{k.lower(): v for k, v in fields.items()}}
        )

    for cohort in ("config_first_1000", "all_requests"):
        sample = text[cohort]
        assert (
            sample["pairs"] == controls[cohort]["exposed_pairs"]
            and sample["qrels_rows"] == controls[cohort]["qrels_rows"]
        )
        for method, metrics in sample["results"].items():
            add(
                "exposure_rerank",
                cohort,
                method,
                requests=sample["requests"],
                **{k: metrics[k] for k in ("MRR@10", "MAP@10", "NDCG@10")},
            )
        for method in ("v1_din", "bge", "bge_pop", "bge_adapter", "bge_adapter_pop", "v2_final"):
            m = controls[cohort]["results"][method]
            add(
                "exposure_rerank",
                cohort,
                method,
                requests=sample["requests"],
                **{
                    "MRR@10": m["official"]["MRR@10"],
                    "MAP@10": m["official"]["MAP@10"],
                    "NDCG@10": m["NDCG@10"],
                },
            )
    base = read(args.run / "base/recall.json")["test"]
    for method, m in {
        **{k: base[k] for k in ("pop", "itemcf", "tower", "rrf")},
        **controls["full_catalog_ablation"]["metrics"],
        **{k: v["metrics"] for k, v in retrieval.items()},
    }.items():
        add(
            "full_catalog",
            "all_requests",
            method,
            requests=m["requests"],
            **{k: m[k] for k in ("recall@10", "recall@50", "recall@100", "hits@10", "hits@50", "hits@100")},
        )
    alignment = read(args.content / "alignment.json")["test"]
    for name, m in (
        ("v1_rrf_din", read(args.run / "base/din.json")["test"]["cascade"]),
        ("v2_exact", alignment["aligned10"]),
        ("v2_ivf", cascade["test"]["final"]),
    ):
        add(
            "end_to_end",
            "all_requests",
            name,
            requests=m["requests"],
            **{k: m[k] for k in ("hits@10", "recall@10", "mrr@10", "map@10", "ndcg@10")},
        )
    for name, bench in latency.items():
        fields = {
            k: bench.get(k)
            for k in (
                "requests",
                "p50_ms",
                "p95_ms",
                "build_seconds",
                "index_files_bytes",
                "host_uss_delta_bytes",
                "device_index_bytes",
            )
        }
        add("index_benchmark", "rank_val_sample", name, **fields)
    for stage, counts in feedback.items():
        add(
            "candidate_feedback",
            stage,
            selected,
            **{k: v for k, v in counts.items() if k != "fit_visibility"},
        )
    result = {
        "dataset_revision": split["revision"],
        "catalog": controls["catalog_size"],
        "test": split["stages"]["test"],
        "frozen_policy": selected,
        "adapter_epoch": controls["adapter_epoch"],
        "text_references": text,
        "ann_quality": retrieval,
        "index_benchmarks": latency,
        "ann_cascade": cascade,
        "feedback": feedback,
        "exact_visibility": archived["slices"],
        "exact_cascade": alignment,
    }
    write_json(args.dest / "evidence.json", result)
    path = args.dest / "experiments.csv"
    tmp = path.with_suffix(".tmp.csv")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        fields = list(dict.fromkeys(k for row in table for k in row))
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    tmp.replace(path)
    print(len(table), "experiment rows; verified", len(rows), "test requests", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "content", "study", "controls", "dest"):
        p.add_argument("--" + key, type=Path, required=True)
    run(p.parse_args())

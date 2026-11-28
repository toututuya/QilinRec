"""Build a V2 serving index, replay its Top 100, or time individual already-encoded requests."""

import argparse
import gc
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
import psutil
import torch
import yaml
from filelock import FileLock

from scripts.align_content import load_scores, replay
from scripts.compare_qilin import content_models
from scripts.content_cascade import run_cascade
from scripts.run_content import profiles, queries, report, slice_metrics
from scripts.verify_run import paired_interval, request_ndcg
from src.data import load_requests
from src.metrics import retrieval_metrics
from src.recall.classical import Popularity
from src.recall.content import ContentIndex, combine
from src.recall.content_ann import ContentFaiss
from src.runtime import ROOT, write_json


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frozen_inputs(run, content):
    data = load_requests(run)
    cfg = json.loads((run / "config.json").read_text())["config"]
    split = json.loads((run / "split.json").read_text())
    policy = json.loads((content / "policy.json").read_text())
    selected = policy["selected"]
    if not selected.startswith("w") or float(selected[1:]) <= 0:
        raise ValueError("This fused index requires a validated positive popularity weight")
    matrix = np.load(content / "articles.npy", mmap_mode="r")
    catalog = np.load(run / "catalog.npy")
    pop = Popularity(data["fit"], len(matrix), split["cutoffs"][0], cfg["pop_half_life_hours"])
    return data, matrix, catalog, pop, np.log1p(pop.scores), policy


def index_signature(args, cfg, prior, policy):
    return {
        "kind": args.engine,
        "config": cfg,
        "faiss": faiss.__version__,
        "policy": policy["selected"],
        "catalog_sha256": sha(args.run / "catalog.npy"),
        "article_vectors_sha256": sha(args.content / "articles.npy"),
        "prior_sha256": hashlib.sha256(prior.tobytes()).hexdigest(),
        "adapter_sha256": sha(args.content / "adapter.best.pt"),
    }


def build(args, cfg, matrix, catalog, pop, prior, policy, signature):
    directory = args.out / args.engine
    with FileLock(str(directory) + ".lock"):
        if directory.exists():
            original = json.loads((directory / "build.json").read_text())
            if original["inputs"] != signature:
                raise ValueError("Existing index belongs to a different release or configuration")
            return
        with tempfile.TemporaryDirectory(dir=args.out, prefix=args.engine + "-") as staging:
            staging = Path(staging)
            started = time.perf_counter()
            engine = ContentFaiss.build(
                args.engine, matrix, catalog, prior, float(policy["selected"][1:]), pop.order, cfg
            )
            seconds = time.perf_counter() - started
            engine.save(staging)
            size = sum(p.stat().st_size for p in staging.iterdir())
            write_json(
                staging / "build.json",
                {
                    "inputs": signature,
                    "indexed_items": int(engine.index.ntotal),
                    "dimension": engine.index.d,
                    "build_seconds": seconds,
                    "save_seconds": time.perf_counter() - started - seconds,
                    "index_files_bytes": size,
                },
            )
            staging.rename(directory)
        print("Built", args.engine, round(seconds, 2), "seconds", size, "bytes", flush=True)


def overlap(pred, exact):
    ratios = []
    for actual, reference in zip(pred, exact, strict=True):
        target = set(reference[reference != 0].tolist())
        ratios.append(
            len(target.intersection(actual.tolist())) / len(target) if target else float(not actual.any())
        )
    return {
        "mean_top100_overlap": float(np.mean(ratios)),
        "identical_ordered_requests": int((pred == exact).all(1).sum()),
    }


def torch_search(matrix, catalog, pop, prior, weight, device):
    full = ContentIndex(matrix, catalog, device)
    warm = ContentIndex(matrix, catalog[prior[catalog] > 0], device)
    available = np.zeros(len(matrix), bool)
    available[catalog] = True

    def search(query, rows, k):
        semantic = full.search(query, rows, k)[0.0]
        warm_result = warm.search(query, rows, k, (weight,), prior)[weight]
        ids = combine(semantic, warm_result, prior, weight)[0]
        for j in np.flatnonzero(~query.any(1)):
            fallback = pop.retrieve(rows[j], k, available)
            ids[j, : len(fallback)] = [i for i, _ in fallback]
        return ids

    return search


def benchmark(args, cfg, rows, q, search, reference, info):
    ids = np.random.default_rng(cfg["seed"]).permutation(len(rows))
    warm, sample = np.split(
        ids[: cfg["warmup"] + cfg["latency_requests"]], [min(cfg["warmup"], len(ids) // 2)]
    )
    if not len(sample):
        raise ValueError("The latency experiment needs at least one held-out query")

    def synchronize():
        if args.engine == "torch" and args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)

    measured = []
    outputs = []
    for j in np.r_[warm, sample]:
        single = q[j : j + 1]
        group = [rows[j]]
        synchronize()
        began = time.perf_counter_ns()
        output = search(single, group, 100)
        synchronize()
        measured.append((time.perf_counter_ns() - began) / 1e6)
        outputs.append(output[0])
    elapsed = np.asarray(measured[len(warm) :])
    info.update(
        {
            "stage": "rank_val",
            "batch_size": 1,
            "warmup": len(warm),
            "requests": len(sample),
            "request_sha256": hashlib.sha256(
                np.asarray([rows[j].rid for j in sample], dtype="<i8").tobytes()
            ).hexdigest(),
            "p50_ms": float(np.quantile(elapsed, 0.50)),
            "p95_ms": float(np.quantile(elapsed, 0.95)),
            "sample_overlap": overlap(np.asarray(outputs[len(warm) :]), reference[sample]),
            "cpu_affinity": psutil.Process().cpu_affinity()
            if hasattr(psutil.Process, "cpu_affinity")
            else None,
            "threads": cfg["threads"],
            "platform": platform.platform(),
            "cpu": next(
                (
                    s.split(":", 1)[1].strip()
                    for s in (
                        Path("/proc/cpuinfo").read_text().splitlines()
                        if Path("/proc/cpuinfo").is_file()
                        else []
                    )
                    if s.startswith("model name")
                ),
                platform.processor(),
            ),
            "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        }
    )
    write_json(args.out / f"{args.engine}.latency.json", info)


def frozen_cascade(args, directory, rows, catalog, policy):
    cfg = json.loads((args.run / "config.json").read_text())["config"]
    alpha = json.loads((args.content / "alignment_policy.json").read_text())["selected_retrieval_weight"]
    write_json(directory / "policy.json", policy)
    run_cascade(argparse.Namespace(run=args.run, out=directory, device=args.device))
    result = {"frozen_retrieval_rank_weight": alpha}
    for stage in ("rank_val", "test"):
        group = rows[stage]
        raw = load_scores(directory, stage, group)
        shortlist, top = replay(raw, range(len(group)), alpha, cfg)
        with np.load(args.content / f"{stage}.aligned.npz") as exact:
            np.testing.assert_array_equal(exact["request_ids"], [r.rid for r in group])
            exact_top = np.split(exact["top_items"], exact["top_offsets"][1:-1])
        result[stage] = {
            "shortlist": retrieval_metrics(shortlist, group, len(catalog), (50,)),
            "final": retrieval_metrics(top, group, len(catalog), (10,)),
            "minus_frozen_exact_ndcg10": paired_interval(
                request_ndcg(group, top) - request_ndcg(group, exact_top), group, cfg["seed"]
            ),
        }
        temp = directory / f"{stage}.aligned.tmp.npz"
        np.savez(
            temp,
            request_ids=raw["request_ids"],
            short_items=np.concatenate(shortlist).astype(np.int32),
            short_offsets=np.r_[0, np.cumsum(list(map(len, shortlist)))],
            top_items=np.concatenate(top).astype(np.int32),
            top_offsets=np.r_[0, np.cumsum(list(map(len, top)))],
        )
        temp.replace(directory / f"{stage}.aligned.npz")
    write_json(args.out / f"{args.engine}.cascade.json", result)


def run(args):
    cfg = yaml.safe_load(args.config.read_text())
    faiss.omp_set_num_threads(cfg["threads"])
    torch.set_num_threads(min(4, cfg["threads"]))
    torch.set_float32_matmul_precision("highest")
    args.out.mkdir(parents=True, exist_ok=True)
    data, matrix, catalog, pop, prior, policy = frozen_inputs(args.run, args.content)
    selected = policy["selected"]
    directory = args.out / args.engine
    if args.engine != "torch":
        signature = index_signature(args, cfg, prior, policy)
        if args.mode == "build":
            return build(args, cfg, matrix, catalog, pop, prior, policy, signature)
        info = json.loads((directory / "build.json").read_text())
        if info["inputs"] != signature:
            raise ValueError("Index signature differs from the frozen model inputs")
        if args.mode == "cascade":
            return frozen_cascade(args, directory, data, catalog, policy)
    elif args.mode != "latency":
        raise ValueError("The production Torch scan already has archived quality results; use latency mode")
    stage = "rank_val" if args.mode == "latency" else args.stage
    rows = data[stage]
    model = content_models(args.content, policy)["bge_adapter"].to(args.device)
    q = queries(model, profiles(args.content, stage, rows), args.device)
    with np.load(args.content / f"{stage}.top100.npz") as original:
        np.testing.assert_array_equal(original["request_ids"], [r.rid for r in rows])
        reference, popular = original[selected], original["pop"]
    gc.collect()
    proc = psutil.Process()
    before = proc.memory_full_info().uss
    started = time.perf_counter()
    if args.engine == "torch":
        use_cuda = args.device.startswith("cuda")
        if use_cuda:
            torch.cuda.synchronize(args.device)
        gpu_before = torch.cuda.memory_allocated(args.device) if use_cuda else 0
        search = torch_search(matrix, catalog, pop, prior, float(selected[1:]), args.device)
        if use_cuda:
            torch.cuda.synchronize(args.device)
        info = {
            "build_seconds": time.perf_counter() - started,
            "device_index_bytes": torch.cuda.memory_allocated(args.device) - gpu_before if use_cuda else None,
        }
    else:
        engine = ContentFaiss.load(directory)
        search = engine.search
        info["load_seconds"] = time.perf_counter() - started
    info["host_uss_delta_bytes"] = proc.memory_full_info().uss - before
    if args.mode == "latency":
        return benchmark(args, cfg, rows, q, search, reference, info)
    batch = cfg["batch_size"]
    started = time.perf_counter()
    pred = np.concatenate(
        [search(q[i : i + batch], rows[i : i + batch], reference.shape[1]) for i in range(0, len(q), batch)]
    )
    elapsed = time.perf_counter() - started
    allowed = np.zeros(len(matrix), bool)
    allowed[catalog] = True
    for r, group in zip(rows, pred, strict=True):
        ids = group[group != 0]
        if len(set(ids.tolist())) != len(ids) or r.seen.intersection(ids.tolist()) or not allowed[ids].all():
            raise ValueError("Duplicate or ineligible result")
    known = np.zeros(len(matrix), bool)
    known[np.load(args.run / "known.npy")] = True
    info.update(
        {
            "stage": stage,
            "batch_size": batch,
            "search_seconds": elapsed,
            **overlap(pred, reference),
            "metrics": report(pred, rows, len(catalog)),
            "fit_visibility": slice_metrics(pred, rows, known),
        }
    )
    dst = directory / f"{stage}.top100.npz"
    temp = dst.with_suffix(".tmp.npz")
    np.savez(temp, request_ids=np.array([r.rid for r in rows]), pop=popular, **{selected: pred})
    temp.replace(dst)
    write_json(args.out / f"{args.engine}.{stage}.json", info)
    print(
        args.engine,
        stage,
        info["metrics"]["hits@100"],
        info["mean_top100_overlap"],
        round(elapsed, 2),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--content", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/content_ann.yaml")
    p.add_argument("--engine", choices=("flat", "ivf", "torch"), required=True)
    p.add_argument("--mode", choices=("build", "retrieve", "latency", "cascade"), required=True)
    p.add_argument("--stage", choices=("rank_val", "test"), default="test")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.out / (args.engine + ".job.lock"))):
        run(args)

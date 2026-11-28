import time

import faiss
import numpy as np
import torch

from scripts.train_recall import make_model
from src.data import load_requests, profile_arrays
from src.metrics import retrieval_metrics
from src.recall.index import batched_search, build_indexes
from src.recall.tower import item_vectors, user_vectors
from src.runtime import configure, evenly_spaced, write_json


def build(args, cfg):
    faiss.omp_set_num_threads(cfg["faiss_threads"])
    run, folder = args.run, args.run / args.variant
    rows = load_requests(run)
    model = make_model(run, cfg, args.variant, args.device)
    ckpt = torch.load(folder / "tower.best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    n = len(model.item.codes)
    ids = np.load(run / "catalog.npy")
    start = time.perf_counter()
    vec = item_vectors(model, np.arange(n, dtype=np.int32), args.device).cpu().numpy()
    np.save(folder / "vectors.npy", vec)
    enc_s = time.perf_counter() - start
    flat, ivf, stats = build_indexes(vec, ids, cfg)
    faiss.write_index(ivf.index, str(folder / "catalog.ivf.index"))
    stats.update(encode_s=enc_s, best_epoch=ckpt["epoch"], threads=cfg["faiss_threads"])
    stages = ("rank_train", "rank_val", "test") if args.variant == "base" else ("test",)
    for stage in stages:
        queries = (
            user_vectors(model, *profile_arrays(rows[stage], cfg["history_length"]), args.device)
            .cpu()
            .numpy()
        )
        np.save(folder / f"{stage}.queries.npy", queries)
    bench = evenly_spaced(len(rows["test"]), cfg["index_benchmark_queries"])
    group = [rows["test"][i] for i in bench]
    sample = queries[bench]
    tops = {}
    for name, index in [("flat", flat), ("ivf", ivf)]:
        index.search(sample[:4], group[:4], cfg["recall_k"])
        start = time.perf_counter()
        top = batched_search(index, sample, group, cfg["recall_k"])
        dt = time.perf_counter() - start
        tops[name] = [[i for i, _ in p] for p in top]
        stats[name] = dict(
            batch_amortized_ms=1000 * dt / len(group),
            queries=len(group),
            batch_size=128,
            **retrieval_metrics(tops[name], group, len(ids), (100,)),
        )
    stats["ivf_flat_overlap100"] = float(
        np.mean([len(set(x) & set(y)) / max(1, len(x)) for x, y in zip(tops["flat"], tops["ivf"])])
    )
    write_json(folder / "faiss.json", stats)
    print(
        args.variant, "index built", stats["ivf_build_s"], "overlap", stats["ivf_flat_overlap100"], flush=True
    )


if __name__ == "__main__":
    args, cfg = configure("Encode the note catalog and build and benchmark FAISS indexes")
    build(args, cfg)

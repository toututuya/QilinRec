"""Replay only the locked V2 exact retrieval, LambdaRank and DIN pipeline from checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from filelock import FileLock
from scipy import sparse
from src.metrics import retrieval_metrics
from src.recall.index import CatalogIndex
from src.runtime import write_json

from scripts.align_content import replay
from scripts.compare_qilin import content_models, ranker_scores
from scripts.content_ann import frozen_inputs, torch_search
from scripts.content_cascade import make_frames
from scripts.run_content import profiles, queries, slice_metrics


def compare_release(content, request_ids, retrieved, shortlist, final, selected):
    with np.load(content / "test.top100.npz") as saved:
        np.testing.assert_array_equal(request_ids, saved["request_ids"])
        same_retrieval = int((retrieved == saved[selected]).all(axis=1).sum())
    with np.load(content / "test.aligned.npz") as saved:
        np.testing.assert_array_equal(request_ids, saved["request_ids"])
        same = {}
        for name, groups in (("short", shortlist), ("top", final)):
            ids = saved[name + "_items"]
            offsets = saved[name + "_offsets"]
            same[name] = sum(
                np.array_equal(group, ids[a:b])
                for group, a, b in zip(groups, offsets[:-1], offsets[1:], strict=True)
            )
    return {"top100": same_retrieval, "top50": same["short"], "top10": same["top"]}


def run(args):
    data, matrix, catalog, pop, prior, policy = frozen_inputs(args.run, args.content)
    cfg = json.loads((args.run / "config.json").read_text())["config"]
    split = json.loads((args.run / "split.json").read_text())
    manifest = json.loads((args.content / "text_manifest.json").read_text())
    if manifest["catalog_sha256"] != hashlib.sha256((args.run / "catalog.npy").read_bytes()).hexdigest():
        raise ValueError("Article vectors and release catalog differ")
    stage = "test"
    rows = data[stage]
    rids = np.array([r.rid for r in rows], dtype="<i8")
    fingerprint = hashlib.sha256(rids.tobytes()).hexdigest()
    if (
        fingerprint != manifest["request_hashes"][stage]
        or fingerprint != split["stages"][stage]["request_id_sha256"]
    ):
        raise ValueError("Request identities differ from the frozen test split")
    faiss.omp_set_num_threads(cfg["faiss_threads"])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    adapter = content_models(args.content, policy)["bge_adapter"].to(args.device)
    q = queries(adapter, profiles(args.content, stage, rows), args.device)
    selected = policy["selected"]
    search = torch_search(matrix, catalog, pop, prior, float(selected[1:]), args.device)
    predicted = search(q, rows, cfg["recall_k"])
    del search
    print("Retrieved", len(rows), "requests over", len(catalog), "catalog entries", flush=True)

    base = args.run / "base"
    available = np.zeros(len(matrix), dtype=bool)
    available[catalog] = True
    groups, frames = make_frames(
        {selected: predicted},
        rows,
        np.load(args.run / "meta.npy", mmap_mode="r"),
        np.load(base / "vectors.npy", mmap_mode="r"),
        np.load(base / "test.queries.npy"),
        pop,
        available,
        sparse.load_npz(base / "itemcf.npz"),
        CatalogIndex(faiss.read_index(str(base / "catalog.ivf.index")), catalog),
        cfg,
    )
    frame = frames[selected]
    frame["prerank"], frame["din"] = ranker_scores(args.run, rows, frame, cfg, split["users"], args.device)
    alpha = json.loads((args.content / "alignment_policy.json").read_text())["selected_retrieval_weight"]
    shortlist, final = replay(frame, range(len(rows)), alpha, cfg)
    known = np.zeros(len(matrix), dtype=bool)
    known[np.load(args.run / "known.npy")] = True
    result = {
        "requests_sha256": fingerprint,
        "catalog_entries": len(catalog),
        "dataset_revision": manifest["dataset_revision"],
        "adapter_epoch": policy["epoch"],
        "retrieval_policy": selected,
        "retrieval_order_weight": alpha,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "retrieval": retrieval_metrics(groups[selected], rows, len(catalog), (10, 50, 100)),
        "shortlist": retrieval_metrics(shortlist, rows, len(catalog), (50,)),
        "final": retrieval_metrics(final, rows, len(catalog), (10,)),
        "visibility": slice_metrics(predicted, rows, known),
    }
    if args.verify_release:
        result["identical_ordered_requests"] = compare_release(
            args.content, rids, predicted, shortlist, final, selected
        )
    arrays = {"request_ids": rids, "top100": predicted}
    for name, lists in (("short", shortlist), ("top", final)):
        arrays[name + "_items"] = np.concatenate(lists).astype(np.int32)
        arrays[name + "_offsets"] = np.r_[0, np.cumsum(list(map(len, lists)))]
    tmp = args.out / "predictions.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(args.out / "predictions.npz")
    write_json(args.out / "metrics.json", result)
    print(
        "Top 100 and final Top 10 hits:",
        result["retrieval"]["hits@100"],
        result["final"]["hits@10"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/release"))
    parser.add_argument("--content", type=Path, default=Path("runs/content-v2"))
    parser.add_argument("--out", type=Path, default=Path("runs/v2-only"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--verify-release",
        action="store_true",
        help="Compare final lists to the frozen release after inference.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.out / ".replay.lock"), timeout=0):
        run(args)

"""Score the locked content-retrieval candidates and equal-budget popularity with the original rankers."""

import argparse
import json
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
import torch
from filelock import FileLock
from scipy import sparse

from scripts.verify_run import paired_interval, request_ndcg
from src.data import load_requests
from src.features import FEATURES, FrameWriter, channel_maps, features, rank_groups
from src.metrics import retrieval_metrics
from src.rank import CTRModel, RankingDataset
from src.recall.classical import Popularity, cf_candidates, cf_scores
from src.recall.index import CatalogIndex, batched_search
from src.runtime import write_json


def make_frames(pools, rows, meta, vectors, q, pop, available, graph, engine, cfg):
    writers = {name: FrameWriter() for name in pools}
    tower = batched_search(engine, q, rows, cfg["recall_k"])
    groups = {name: [] for name in pools}
    for j, (r, neighbors) in enumerate(zip(rows, tower, strict=True)):
        cf = cf_scores(graph, r.hist)
        maps = channel_maps(
            [
                pop.retrieve(r, cfg["recall_k"], available),
                cf_candidates(cf, r.seen, available, cfg["recall_k"]),
                neighbors,
            ]
        )
        candidates = {name: group[j][group[j] != 0] for name, group in pools.items()}
        all_ids = np.unique(np.concatenate(list(candidates.values())))
        if not available[all_ids].all() or set(all_ids.tolist()).intersection(r.seen):
            raise ValueError("Candidate ineligible at request time")
        x = features(r, all_ids, maps, cf, pop, meta, vectors, q[j], cfg["rrf_offset"])
        for name, ids in candidates.items():
            if len(np.unique(ids)) != len(ids) or len(ids) > cfg["recall_k"]:
                raise ValueError("Duplicate candidate or budget exceeded")
            writers[name].add(ids, x[np.searchsorted(all_ids, ids)])
            groups[name].append(ids.tolist())
    frames = {
        name: {
            "x": np.concatenate(w.x),
            "items": np.concatenate(w.items),
            "offsets": np.r_[0, np.cumsum(w.sizes)],
        }
        for name, w in writers.items()
    }
    return groups, frames


def run_cascade(args):
    run, out, device = args.run, args.out, args.device
    base = run / "base"
    policy = json.loads((out / "policy.json").read_text())
    selected = policy["selected"]
    cfg = json.loads((run / "config.json").read_text())["config"]
    split = json.loads((run / "split.json").read_text())
    data = load_requests(run)
    catalog = np.load(run / "catalog.npy")
    meta = np.load(run / "meta.npy", mmap_mode="r")
    vectors = np.load(base / "vectors.npy")
    codes = np.load(run / "codes.npy")
    faiss.omp_set_num_threads(cfg["faiss_threads"])
    torch.set_num_threads(4)
    old_tower = CatalogIndex(faiss.read_index(str(base / "catalog.ivf.index")), catalog)
    available = np.zeros(len(codes), bool)
    available[catalog] = True
    pop = Popularity(data["fit"], len(codes), split["cutoffs"][0], cfg["pop_half_life_hours"])
    graph = sparse.load_npz(base / "itemcf.npz")
    pre = lgb.Booster(model_file=str(base / "lambdarank.best.txt"))
    scale = np.load(base / "rank_feature_scale.npy")
    rankers = {}
    for name in ("mlp", "din"):
        net = CTRModel(name, vectors, codes, split["users"], cfg, len(FEATURES)).to(device)
        net.load_state_dict(
            torch.load(base / f"{name}.best.pt", map_location=device, weights_only=True)["model"]
        )
        rankers[name] = net.eval()
    outputs = {"selected": selected, "ranker_checkpoints": "frozen base release"}
    for stage in ("rank_val", "test"):
        rows = data[stage]
        with np.load(out / f"{stage}.top100.npz") as f:
            rids = f["request_ids"]
            np.testing.assert_array_equal(rids, [r.rid for r in rows])
            pools = {name: f[name] for name in dict.fromkeys(("pop", selected))}
        raw, frames = make_frames(
            pools,
            rows,
            meta,
            vectors,
            np.load(base / f"{stage}.queries.npy"),
            pop,
            available,
            graph,
            old_tower,
            cfg,
        )
        stats, ndcg, recs = {}, {}, {}
        for method, frame in frames.items():
            stats[method] = {"retrieval": retrieval_metrics(raw[method], rows, len(catalog), (100,))}
            preds = pre.predict(frame["x"], num_threads=cfg["faiss_threads"])
            if method == selected:
                full = RankingDataset(rows, frame, cfg, scale, device, labeled=False)
                dst = out / f"{stage}.model_scores.npz"
                tmp = dst.with_suffix(".tmp.npz")
                np.savez(
                    tmp,
                    request_ids=rids,
                    items=frame["items"],
                    offsets=frame["offsets"],
                    prerank=preds,
                    din=full.predict(rankers["din"], cfg["rank_batch"]),
                )
                tmp.replace(dst)
                del full
            shortlist, take = rank_groups(preds, frame["items"], frame["offsets"], cfg["prerank_k"])
            offsets = np.r_[0, np.cumsum([len(g) for g in shortlist])]
            short = {"take": take, "offsets": offsets}
            stats[method]["lambdarank"] = retrieval_metrics(shortlist, rows, len(catalog), (50,))
            prediction = {
                "request_ids": rids,
                "shortlist_items": frame["items"][take],
                "shortlist_offsets": offsets,
                "shortlist_scores": preds[take],
            }
            batch = RankingDataset(rows, frame, cfg, scale, device, labeled=False, shortlist=short)
            for name, net in rankers.items():
                logits = batch.predict(net, cfg["rank_batch"])
                top, ix = rank_groups(logits, batch.ids, batch.offsets, cfg["rank_k"])
                stats[method][name] = retrieval_metrics(top, rows, len(catalog), (10,))
                prediction[name + "_items"] = batch.ids[ix]
                prediction[name + "_offsets"] = np.r_[0, np.cumsum([len(g) for g in top])]
                prediction[name + "_scores"] = logits[ix]
                if name == "din":
                    ndcg[method] = request_ndcg(rows, top)
            dst = out / f"{stage}.{method}.cascade.npz"
            tmp = dst.with_suffix(".tmp.npz")
            np.savez(tmp, **prediction)
            tmp.replace(dst)
            recs[method] = np.array(
                [
                    len(set(r.positives.tolist()).intersection(g)) / max(1, len(r.positives))
                    for r, g in zip(rows, raw[method], strict=True)
                ]
            )
        bootstrap = paired_interval(recs[selected] - recs["pop"], rows, cfg["seed"])
        bootstrap["delta_macro_recall100"] = bootstrap.pop("delta_ndcg10")
        outputs[stage] = {
            "policies": stats,
            "selected_minus_pop_recall100": bootstrap,
            "selected_minus_pop_din_ndcg10": paired_interval(ndcg[selected] - ndcg["pop"], rows, cfg["seed"]),
        }
        print(
            stage,
            selected,
            {
                n: stats[selected][n][f"hits@{k}"]
                for n, k in (("retrieval", 100), ("lambdarank", 50), ("mlp", 10), ("din", 10))
            },
            flush=True,
        )
    write_json(out / "cascade.json", outputs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    with FileLock(args.out / ".content.lock", timeout=0):
        run_cascade(args)

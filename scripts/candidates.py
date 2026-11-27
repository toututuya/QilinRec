import json

import faiss
import numpy as np
from scipy import sparse

from src.data import load_requests
from src.features import FrameWriter, channel_maps, features
from src.metrics import retrieval_metrics
from src.recall.classical import Popularity, build_itemcf, cf_candidates, cf_scores
from src.recall.fusion import fuse
from src.recall.index import CatalogIndex, batched_search
from src.runtime import configure, write_json

METHODS = {
    "pop": (0,),
    "itemcf": (1,),
    "tower": (2,),
    "cf+tower": (1, 2),
    "rrf": (0, 1, 2),
    "no_itemcf": (0, 2),
    "no_tower": (0, 1),
}


def generate(args, cfg):
    run, folder = args.run, args.run / args.variant
    data = load_requests(run)
    catalog = np.load(run / "catalog.npy")
    faiss.omp_set_num_threads(cfg["faiss_threads"])
    engine = CatalogIndex(faiss.read_index(str(folder / "catalog.ivf.index")), catalog)
    k, offset = cfg["recall_k"], cfg["rrf_offset"]
    if args.variant != "base":
        q = np.load(folder / "test.queries.npy")
        found = batched_search(engine, q, data["test"], k)
        top = [[i for i, _ in row] for row in found]
        write_json(
            folder / "recall.json", {"test": {"tower": retrieval_metrics(top, data["test"], len(catalog))}}
        )
        return
    meta = np.load(run / "meta.npy", mmap_mode="r")
    vectors = np.load(folder / "vectors.npy", mmap_mode="r")
    cutoff = json.loads((run / "split.json").read_text())["cutoffs"][0]
    pop = Popularity(data["fit"], len(meta), cutoff, cfg["pop_half_life_hours"])
    graph_path = folder / "itemcf.npz"
    graph = build_itemcf(data["fit"], len(meta), cutoff, cfg)
    sparse.save_npz(graph_path, graph)
    available = np.zeros(len(meta), bool)
    available[catalog] = True
    observed = np.intersect1d(np.load(run / "known.npy"), catalog)
    known_mask = np.zeros(len(meta), bool)
    known_mask[observed] = True
    metrics = {}
    print("ItemCF", graph.nnz, "edges;", len(observed), "previously observed catalog items", flush=True)
    for stage in ["rank_train", "rank_val", "test"]:
        rows = data[stage]
        q = np.load(folder / f"{stage}.queries.npy")
        found = batched_search(engine, q, rows, k)
        conservative = None
        if stage == "test":
            other = faiss.IndexFlatIP(vectors.shape[1])
            other.add(np.ascontiguousarray(vectors[observed]))
            conservative = batched_search(CatalogIndex(other, observed), q, rows, k)
        candidate, exposures = FrameWriter(), FrameWriter()
        rankings = {name: [] for name in METHODS}
        known_ranks = []
        for j, (r, tower) in enumerate(zip(rows, found)):
            cf = cf_scores(graph, r.hist)
            pop_top = pop.retrieve(r, k, available)
            cf_top = cf_candidates(cf, r.seen, available, k)
            channels = [pop_top, cf_top, tower]
            lists = {name: fuse([channels[c] for c in ix], offset, k) for name, ix in METHODS.items()}
            for name, top in lists.items():
                rankings[name].append(top)
            ids = np.array(lists["rrf"], np.int32)
            if len(set(ids).intersection(r.seen)):
                raise AssertionError("Previously interacted item in candidate list")
            maps = channel_maps(channels)
            for frame, items in [(candidate, ids), (exposures, r.items)]:
                x = features(r, items, maps, cf, pop, meta, vectors, q[j], offset)
                frame.add(items, x)
            if conservative is not None:
                ks = [
                    pop.retrieve(r, k, known_mask),
                    cf_candidates(cf, r.seen, known_mask, k),
                    conservative[j],
                ]
                known_ranks.append(fuse(ks, offset, k))
        metrics[stage] = {name: retrieval_metrics(top, rows, len(catalog)) for name, top in rankings.items()}
        if conservative is not None:
            metrics["test_observed_catalog"] = {
                "rrf_exact_tower": retrieval_metrics(known_ranks, rows, len(observed))
            }
        candidate.save(folder / f"{stage}.candidates.npz")
        exposures.save(folder / f"{stage}.exposures.npz")
        print(stage, "generated", len(rows), "requests", flush=True)
    write_json(folder / "recall.json", metrics)


if __name__ == "__main__":
    args, cfg = configure("Fuse retrieval channels and construct exposure and serving feature frames")
    generate(args, cfg)

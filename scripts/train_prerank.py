import json

import numpy as np

from src.data import load_requests
from src.features import rank_groups
from src.metrics import exposure_metrics, retrieval_metrics
from src.prerank import fit_prerank, load_frame
from src.runtime import configure, write_json


def train(args, cfg):
    run, folder = args.run, args.run / "base"
    rows = load_requests(run)
    frames = [load_frame(folder, stage, "exposures") for stage in ("rank_train", "rank_val")]
    labels = [np.concatenate([r.y for r in rows[s]]) for s in ("rank_train", "rank_val")]
    model = fit_prerank(folder, *frames, *labels, cfg)
    n = len(np.load(run / "catalog.npy"))
    report = {}
    for stage in ("rank_val", "test"):
        r = rows[stage]
        exposure = load_frame(folder, stage, "exposures")
        candidate = load_frame(folder, stage, "candidates")
        pr = model.predict(exposure["x"], num_threads=cfg["faiss_threads"])
        scores = model.predict(candidate["x"], num_threads=cfg["faiss_threads"])
        metrics = {}
        for name, es, cs in [
            ("rrf", exposure["x"][:, -1], candidate["x"][:, -1]),
            ("lambdarank", pr, scores),
        ]:
            metrics[name] = {"exposure": exposure_metrics(es, r, exposure["items"], exposure["offsets"])}
            ranked, ix = rank_groups(cs, candidate["items"], candidate["offsets"], cfg["prerank_k"])
            metrics[name]["cascade"] = retrieval_metrics(ranked, r, n, (10, cfg["prerank_k"]))
            if name == "lambdarank":
                np.savez(
                    folder / f"{stage}.shortlist.npz",
                    take=ix,
                    offsets=np.r_[0, np.cumsum([len(g) for g in ranked])],
                    request_ids=np.array([row.rid for row in r]),
                    items=candidate["items"][ix],
                    scores=cs[ix],
                )
        report[stage] = metrics
    log = json.loads((folder / "lgb_train.json").read_text())
    selection = log["validation"][log["best_iteration"] - 1]["val_ndcg10"]
    np.testing.assert_allclose(selection, report["rank_val"]["lambdarank"]["exposure"]["ndcg@10"], atol=1e-8)
    write_json(folder / "prerank.json", report)
    print("Pre-rank", report["test"]["lambdarank"]["exposure"], flush=True)


if __name__ == "__main__":
    args, cfg = configure("Train request-group LambdaRank on logged exposure feedback")
    train(args, cfg)

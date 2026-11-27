"""Count the logged supervision for each request's frozen V2 Top 100."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.compare_qilin import content_models
from scripts.content_ann import frozen_inputs
from scripts.run_content import slice_metrics, stage_pools
from src.recall.content import ContentIndex
from src.runtime import write_json


def feedback(rows, groups):
    counts = Counter(
        {
            "candidates": 0,
            "clicked": 0,
            "exposed_nonclick": 0,
            "unknown": 0,
            "mixed_requests": 0,
            "pairwise_preferences": 0,
            "requests_with_feedback": 0,
        }
    )
    users = set()
    for r, group in zip(rows, groups, strict=True):
        ids = group[group != 0]
        if len(set(ids.tolist())) != len(ids) or r.seen.intersection(ids.tolist()):
            raise ValueError("Candidate duplication or seen item")
        labels = dict(zip(r.items.tolist(), r.y.tolist(), strict=True))
        observed = [labels[i] for i in ids.tolist() if i in labels]
        positive = sum(observed)
        negative = len(observed) - positive
        counts.update(
            candidates=len(ids),
            clicked=positive,
            exposed_nonclick=negative,
            unknown=len(ids) - len(observed),
            requests_with_feedback=int(bool(observed)),
            mixed_requests=int(positive > 0 and negative > 0),
            pairwise_preferences=positive * negative,
        )
        if positive and negative:
            users.add(r.uid)
    assert counts["candidates"] == counts["clicked"] + counts["exposed_nonclick"] + counts["unknown"]
    return {
        "requests": len(rows),
        "all_exposures": sum(len(r.items) for r in rows),
        "all_clicks": sum(len(r.positives) for r in rows),
        "mixed_users": len(users),
        **counts,
    }


def run(args):
    data, matrix, catalog, pop, prior, policy = frozen_inputs(args.run, args.content)
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    model = content_models(args.content, policy)["bge_adapter"].to(args.device)
    selected = policy["selected"]
    index = ContentIndex(matrix, catalog, args.device)
    warm = ContentIndex(matrix, catalog[prior[catalog] > 0], args.device)
    available = np.zeros(len(matrix), bool)
    available[catalog] = True
    known = np.zeros(len(matrix), bool)
    known[np.load(args.run / "known.npy")] = True
    earlier = json.loads((args.content / "proposed.json").read_text())["rank_train"]
    final = json.loads((args.content / "test.json").read_text())["metrics"]
    out = {}
    for stage, reference in (("rank_train", earlier), ("rank_val", policy["rank_val"]), ("test", final)):
        rows = data[stage]
        if stage == "rank_train":
            pools = stage_pools(
                argparse.Namespace(out=args.content, device=args.device),
                {"k": 100},
                stage,
                rows,
                index,
                warm,
                model,
                pop,
                available,
                prior,
                [float(selected[1:])],
            )
            groups = pools[selected]
        else:
            with np.load(args.content / f"{stage}.top100.npz") as f:
                np.testing.assert_array_equal(f["request_ids"], [r.rid for r in rows])
                groups = f[selected]
        counts = feedback(rows, groups)
        if counts["clicked"] != reference[selected]["hits@100"]:
            raise ValueError("Replayed hits differ from the frozen release")
        counts["fit_visibility"] = slice_metrics(groups, rows, known)
        out[stage] = counts
        print(stage, counts, flush=True)
    write_json(args.out, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "content", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())

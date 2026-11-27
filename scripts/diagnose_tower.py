"""Paired validation-only history and training-duration experiment for the released two-tower."""

import argparse
import json
import os
import pickle
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from scripts.train_recall import make_model, train
from src.data import load_requests, profile_arrays
from src.recall.tower import negative_mask, validation
from src.runtime import write_json


def history_view(rows, mode, length):
    if mode == "reconstructed":
        return rows
    # The paper defines this snapshot as clicks before the request. Keep the
    # original vocabulary, seen filter and all other inputs fixed in this arm.
    return [replace(r, hist=tuple(r.recent[:length])) for r in rows]


def link_input(source, destination):
    if not destination.exists():
        os.link(source, destination)


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


@torch.inference_mode()
def sampled_accuracy(model, rows, cfg, device, seed):
    """Same deterministic clicked pairs and filtered in-batch alternatives across arms."""
    model.eval()
    qi = np.array([j for j, r in enumerate(rows) for _ in r.positives])
    targets = np.concatenate([r.positives for r in rows])
    u, h = profile_arrays(rows, cfg["history_length"])
    rng = np.random.default_rng(seed)
    width = cfg["batch_size"]
    pairs = rng.choice(len(qi), size=min(12, len(qi) // width) * width, replace=False)
    by_batch = []
    for ix in pairs.reshape(-1, width):
        ids, q = targets[ix], qi[ix]
        mask = negative_mask(ids, [rows[j] for j in q])
        logits = (
            model.query(torch.as_tensor(u[q], device=device), torch.as_tensor(h[q], device=device))
            @ model.item(torch.as_tensor(ids, device=device)).T
        )
        logits = (logits / cfg["temperature"]).masked_fill(torch.as_tensor(mask, device=device), -1e9)
        label = torch.arange(width, device=device)
        by_batch.append(
            [
                F.cross_entropy(logits, label).item(),
                (logits.argmax(1) == label).float().mean().item(),
                float(np.mean(1.0 / (~mask).sum(1))),
            ]
        )
    loss, top1, chance = np.mean(by_batch, axis=0)
    return {"pairs": len(pairs), "loss": float(loss), "top1": float(top1), "uniform_top1": float(chance)}


def measure(model, rows, cfg, catalog, device):
    recall, hits = validation(model, profile_arrays(rows, cfg["history_length"]), rows, catalog, device, 100)
    return {
        "requests": len(rows),
        "positives": sum(int(r.y.sum()) for r in rows),
        "recall100": recall,
        "hits100": hits,
    }


def experiment(args):
    cfg = json.loads((args.run / "config.json").read_text())["config"]
    original_seed = cfg["seed"]
    epochs = 12
    seeds = [original_seed + i for i in range(3)]
    cfg.update(tower_epochs=epochs, patience=epochs, val_queries=0)
    data = load_requests(args.run)
    catalog = np.load(args.run / "catalog.npy")
    args.work.mkdir(parents=True, exist_ok=True)
    design = {
        "epochs": epochs,
        "seeds": seeds,
        "selection": "all tower_val requests, exact full-catalog macro Recall@100",
        "later_window": "rank_val",
        "modes": ["reconstructed", "official_recent"],
        "split": json.loads((args.run / "split.json").read_text())["stages"]["fit"]["request_id_sha256"],
    }
    manifest = args.work / "design.json"
    if manifest.exists() and json.loads(manifest.read_text()) != design:
        raise ValueError("Diagnostic directory contains a different experiment")
    write_json(manifest, design)
    result = {"design": design, "trials": []}
    for mode in design["modes"]:
        root = args.work / mode
        root.mkdir(exist_ok=True)
        rows = {
            k: history_view(data[k], mode, cfg["history_length"]) for k in ("fit", "tower_val", "rank_val")
        }
        # The training subprocess interface reads this two-window input only.
        with (root / "requests.pkl").open("wb") as stream:
            pickle.dump({k: rows[k] for k in ("fit", "tower_val")}, stream, protocol=5)
        for seed in seeds:
            run = root / str(seed)
            run.mkdir(exist_ok=True)
            for name in ("codes.npy", "meta.npy", "catalog.npy", "split.json"):
                link_input(args.run / name, run / name)
            link_input(root / "requests.pkl", run / "requests.pkl")
            seed_training(seed)
            cfg["seed"] = seed
            train(SimpleNamespace(run=run, variant="base", device=args.device), cfg)
            curve = json.loads((run / "base/tower_train.json").read_text())
            model = make_model(run, cfg, "base", args.device)
            entry = {"mode": mode, "seed": seed, "curve": curve}
            for tag in ("best", "last"):
                state = torch.load(
                    run / "base" / f"tower.{tag}.pt", map_location=args.device, weights_only=False
                )
                model.load_state_dict(state["model"])
                entry[tag] = {
                    "epoch": state["epoch"] if tag == "best" else len(curve),
                    "later": measure(model, rows["rank_val"], cfg, catalog, args.device),
                    "fit_inbatch": sampled_accuracy(model, rows["fit"], cfg, args.device, 912),
                    "validation_inbatch": sampled_accuracy(model, rows["tower_val"], cfg, args.device, 913),
                }
            result["trials"].append(entry)
            write_json(args.work / "progress.json", result)
            print(mode, seed, "later", entry["best"]["later"], flush=True)
    write_json(args.out, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    experiment(parser.parse_args())

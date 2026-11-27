"""Fit the text user tower, lock a fusion policy on earlier windows, then evaluate the cascade."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from filelock import FileLock
from torch.nn import functional as F

from src.data import load_requests
from src.metrics import retrieval_metrics
from src.recall.classical import Popularity
from src.recall.content import ContentIndex, InterestAdapter, combine
from src.recall.tower import negative_mask
from src.runtime import ROOT, save_torch, write_json


def profiles(out, stage, rows):
    with np.load(out / f"{stage}.profiles.npz") as f:
        np.testing.assert_array_equal(f["request_ids"], [r.rid for r in rows])
        return f["vectors"]


def report(top, rows, catalog_size, ks=(10, 50, 100)):
    return retrieval_metrics([[int(i) for i in group if i] for group in top], rows, catalog_size, ks)


@torch.inference_mode()
def queries(model, encoded, device):
    model.eval()
    return torch.cat(
        [
            model(torch.as_tensor(batch, device=device)).cpu()
            for batch in np.array_split(encoded, max(1, len(encoded) // 2048))
        ]
    ).numpy()


def train(args, cfg, rows, matrix, index, model):
    out, device = args.out, args.device
    fit = rows["fit"]
    train_x = profiles(out, "fit", fit)
    val_x = profiles(out, "tower_val", rows["tower_val"])
    k = cfg["k"]
    clicked = np.concatenate([r.positives for r in fit])
    qr = np.repeat(np.arange(len(fit)), [len(r.positives) for r in fit])
    usable = train_x.any(1)[qr] & matrix[clicked].any(1)
    clicked, qr = clicked[usable], qr[usable]
    rng_seed = cfg["seed"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["adapter_lr"])
    history, best = [], -1.0
    path = out / "adapter.last.pt"
    if path.exists():
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best = state["history"], state["best"]
    for epoch in range(len(history), cfg["adapter_epochs"] + 1):
        start = time.perf_counter()
        loss_sum = 0.0
        if epoch:
            model.train()
            order = np.random.default_rng(rng_seed + epoch).permutation(len(qr))
            for lo in range(0, len(order), cfg["batch_size"]):
                ix = order[lo : lo + cfg["batch_size"]]
                target, q = clicked[ix], qr[ix]
                mask = negative_mask(target, [fit[j] for j in q])
                u = model(torch.as_tensor(train_x[q], device=device))
                v = F.normalize(torch.as_tensor(matrix[target], device=device).float(), dim=1)
                logits = (u @ v.T / cfg["temperature"]).masked_fill(
                    torch.as_tensor(mask, device=device), -1e9
                )
                loss = F.cross_entropy(logits, torch.arange(len(q), device=device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(q)
        ranking = index.search(queries(model, val_x, device), rows["tower_val"], k)[0.0]
        m = report(ranking[0], rows["tower_val"], len(index.ids), (k,))
        history.append(
            {
                "epoch": epoch,
                "fit_loss": loss_sum / len(qr) if epoch else None,
                "fit_pairs": len(qr),
                "validation": m,
                "seconds": time.perf_counter() - start,
            }
        )
        if m[f"recall@{k}"] > best:
            best = m[f"recall@{k}"]
            save_torch(out / "adapter.best.pt", {"model": model.state_dict(), "epoch": epoch, "recall": best})
        save_torch(
            path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best": best,
            },
        )
        write_json(out / "training.json", history)
        print("epoch", epoch, "validation", m[f"hits@{k}"], m[f"recall@{k}"], flush=True)
    write_json(out / "training.json", history)


def stage_pools(args, cfg, stage, rows, index, warm, model, popular, catalog_mask, prior, weights):
    x = profiles(args.out, stage, rows)
    q = queries(model, x, args.device)
    k = cfg["k"]
    s = index.search(q, rows, k)[0.0]
    dense_weights = tuple(w for w in weights if w > 0)
    wtop = warm.search(q, rows, k, dense_weights, prior) if dense_weights else {}
    p = np.zeros_like(s[0])
    for j, r in enumerate(rows):
        top = popular.retrieve(r, k, catalog_mask)
        p[j, : len(top)] = [i for i, _ in top]
    empty = ~x.any(1)
    pools = {"pop": p, "semantic": s[0]}
    for w in weights:
        pick = combine(s, wtop[w], prior, w)[0] if w else s[0].copy()
        pick[empty] = p[empty]
        pools[f"w{w:g}"] = pick
    return pools


def slice_metrics(ids, rows, known):
    slices = {"seen_in_fit": [0, 0], "unseen_in_fit": [0, 0]}
    for r, pred in zip(rows, ids, strict=True):
        picked = set(pred.tolist())
        for i in r.positives.tolist():
            v = slices["seen_in_fit" if known[i] else "unseen_in_fit"]
            v[0] += i in picked
            v[1] += 1
    return {s: {"hits": int(h), "positives": n} for s, (h, n) in slices.items()}


def save_stage(out, stage, pools, names, rows):
    dst = out / f"{stage}.top100.npz"
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, request_ids=np.array([r.rid for r in rows]), **{n: pools[n] for n in names})
    tmp.replace(dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/content.yaml")
    parser.add_argument("--phase", choices=("train", "select", "test", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with FileLock(args.out / ".content.lock", timeout=0):
        run(args)


def run(args):
    cfg = yaml.safe_load(args.config.read_text())
    frozen = json.loads((args.out / "text_manifest.json").read_text())
    if not (args.out / "text_ready.json").is_file():
        raise ValueError("Finish text encoding first")
    if (
        any(cfg[k] != frozen[k] for k in ("dimension", "recent_titles", "profile_tokens", "article_tokens"))
        or cfg["revision"] != frozen["encoder_revision"]
    ):
        raise ValueError("Content configuration and encoded text differ")
    split = json.loads((args.run / "split.json").read_text())
    if (
        frozen["request_hashes"] != {k: v["request_id_sha256"] for k, v in split["stages"].items()}
        or frozen["catalog_sha256"] != hashlib.sha256((args.run / "catalog.npy").read_bytes()).hexdigest()
    ):
        raise ValueError("Base run and encoded text differ")
    run_cfg = json.loads((args.run / "config.json").read_text())["config"]
    if cfg["k"] != 100 or cfg["k"] != run_cfg["recall_k"]:
        raise ValueError("The comparison requires the same 100-candidate budget as the base run")
    if any(w < 0 for w in cfg["pop_weights"]):
        raise ValueError("Warm-subset fusion requires nonnegative weights")
    design = {
        "config": cfg,
        "base_config": run_cfg,
        "text": frozen,
        "epoch_selection": "tower_val macro Recall@100 including frozen epoch zero",
        "weight_selection": "rank_train macro Recall@100, ties prefer popularity",
        "confirmation": "rank_val macro recall must exceed both pop and base RRF; hits must match or exceed both",
        "fallback": "pop",
        "labels": "fit clicks for adapter; logged positives retained in every evaluation",
    }
    design_path = args.out / "design.json"
    if design_path.exists() and json.loads(design_path.read_text()) != design:
        raise ValueError("Select a new directory for a changed experiment")
    write_json(design_path, design)
    rows = load_requests(args.run)
    catalog = np.load(args.run / "catalog.npy")
    matrix = np.load(args.out / "articles.npy", mmap_mode="r")
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    model = InterestAdapter(cfg["dimension"], cfg["adapter_width"]).to(args.device)
    index = ContentIndex(matrix, catalog, args.device)
    if args.phase in ("train", "all"):
        train(args, cfg, rows, matrix, index, model)
    if args.phase == "train":
        return
    state = torch.load(args.out / "adapter.best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(state["model"])
    pop = Popularity(rows["fit"], len(matrix), split["cutoffs"][0], run_cfg["pop_half_life_hours"])
    available = np.zeros(len(matrix), bool)
    available[catalog] = True
    known = np.zeros(len(matrix), bool)
    known[np.load(args.run / "known.npy")] = True
    prior = np.log1p(pop.scores).astype(np.float32)
    warm = ContentIndex(matrix, catalog[prior[catalog] > 0], args.device)

    def evaluate(stage, weights):
        pools = stage_pools(args, cfg, stage, rows[stage], index, warm, model, pop, available, prior, weights)
        metrics = {n: report(p, rows[stage], len(catalog)) for n, p in pools.items()}
        return pools, metrics

    if args.phase in ("select", "all"):
        _, metrics = evaluate("rank_train", cfg["pop_weights"])
        options = ["pop"] + [f"w{w:g}" for w in cfg["pop_weights"]]
        proposed = max(options, key=lambda n: metrics[n]["recall@100"])
        write_json(
            args.out / "proposed.json", {"epoch": state["epoch"], "proposed": proposed, "rank_train": metrics}
        )
        weights = [] if proposed == "pop" else [float(proposed[1:])]
        val, vm = evaluate("rank_val", weights)
        earlier = json.loads((args.run / "base/recall.json").read_text())["rank_val"]
        if vm["pop"]["hits@100"] != earlier["pop"]["hits@100"]:
            raise ValueError("Popularity control differs from the release")
        m = vm[proposed]
        passed = proposed != "pop" and all(
            m["recall@100"] > earlier[n]["recall@100"] and m["hits@100"] >= earlier[n]["hits@100"]
            for n in ("pop", "rrf")
        )
        chosen = proposed if passed else "pop"
        policy = {
            "proposed": proposed,
            "selected": chosen,
            "confirmed": passed,
            "epoch": state["epoch"],
            "rank_val": vm,
            "rank_val_base_rrf": earlier["rrf"],
            "rank_val_slices": {n: slice_metrics(v, rows["rank_val"], known) for n, v in val.items()},
        }
        save_stage(args.out, "rank_val", val, {"pop", chosen}, rows["rank_val"])
        write_json(args.out / "policy.json", policy)
        print("locked", chosen, "rank_val", vm[chosen]["hits@100"], flush=True)
    if args.phase in ("test", "all"):
        policy = json.loads((args.out / "policy.json").read_text())
        if state["epoch"] != policy["epoch"]:
            raise ValueError("Locked policy and selected user tower do not match")
        name = policy["selected"]
        weights = [] if name == "pop" else [float(name[1:])]
        pools, metrics = evaluate("test", weights)
        save_stage(args.out, "test", pools, {"pop", name}, rows["test"])
        write_json(
            args.out / "test.json",
            {
                "selected": name,
                "epoch": state["epoch"],
                "metrics": metrics,
                "slices": {n: slice_metrics(v, rows["test"], known) for n, v in pools.items()},
            },
        )
        print("test", name, metrics[name]["hits@100"], metric_summary(metrics[name]), flush=True)


def metric_summary(metrics):
    return {k: metrics[k] for k in ("recall@100", "retained@100", "coverage@100")}


if __name__ == "__main__":
    main()

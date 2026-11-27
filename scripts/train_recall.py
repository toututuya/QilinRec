import json
import time

import numpy as np
import torch

from src.data import load_requests, profile_arrays
from src.recall.tower import TwoTower, contrastive_loss, negative_mask, validation
from src.runtime import configure, evenly_spaced, save_torch, write_json


def make_model(run, cfg, variant, device):
    meta, codes = [np.load(run / f"{f}.npy") for f in ("meta","codes")]
    users = json.loads((run / "split.json").read_text())["users"]
    return TwoTower(meta, codes, users, cfg, variant).to(device)


def train(args, cfg):
    rows = load_requests(args.run)
    fit = rows["fit"]
    qr = np.array([i for i,r in enumerate(fit) for _ in r.positives], dtype=np.int32)
    target = np.concatenate([r.positives for r in fit])
    user, hist = profile_arrays(fit, cfg["history_length"])
    vrows = [rows["tower_val"][i] for i in evenly_spaced(len(rows["tower_val"]), cfg["val_queries"])]
    vp = profile_arrays(vrows, cfg["history_length"])
    catalog = np.load(args.run / "catalog.npy")
    model = make_model(args.run, cfg, args.variant, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["tower_lr"])
    folder = args.run / args.variant
    folder.mkdir(exist_ok=True)
    last, best = folder / "tower.last.pt", folder / "tower.best.pt"
    history, score, stale = [], -1., 0
    if last.exists():
        state = torch.load(last, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, score, stale = state["history"], state["best"], state["stale"]
    to_device = lambda x: torch.as_tensor(x, device=args.device)
    for epoch in range(len(history), cfg["tower_epochs"]):
        if stale >= cfg["patience"]:
            break
        begin = time.perf_counter()
        rng = np.random.default_rng(cfg["seed"] + epoch)
        torch.manual_seed(cfg["seed"] + epoch)
        order = rng.permutation(len(qr))
        model.train()
        loss_sum = 0.
        for lo in range(0, len(qr), cfg["batch_size"]):
            sel = order[lo:lo+cfg["batch_size"]]
            ids, qids = target[sel], qr[sel]
            group = [fit[i] for i in qids]
            mask = negative_mask(ids, group)
            extra = None
            if args.variant == "exposed_neg":
                extras = []
                for r in group:
                    allowed = [i for i,y in zip(r.items.tolist(), r.y.tolist()) if not y and i not in r.seen]
                    extras.append(rng.choice(allowed) if allowed else 0)
                extra = to_device(np.array(extras))
            loss = contrastive_loss(model, to_device(user[qids]), to_device(hist[qids]), to_device(ids),
                                    to_device(mask), cfg["temperature"], extra)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(sel)
        metric, hits = validation(model, vp, vrows, catalog, args.device, cfg["recall_k"])
        entry = dict(epoch=epoch+1, loss=loss_sum/len(qr), val_recall100=metric, val_hits=hits,
                     val_requests=len(vrows), seconds=time.perf_counter()-begin)
        history.append(entry)
        print(args.variant, entry, flush=True)
        if metric > score:
            score, stale = metric, 0
            save_torch(best, {"model": model.state_dict(), "epoch": epoch+1, "metric": metric})
        else:
            stale += 1
        save_torch(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                              history=history, best=score, stale=stale))
    write_json(folder / "tower_train.json", history)


if __name__ == "__main__":
    args, cfg = configure("Train the PyTorch retrieval model with point-in-time profiles")
    train(args, cfg)

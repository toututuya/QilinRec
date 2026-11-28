"""Interrupt each trainer after its first checkpoint and resume on the smoke data."""

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import torch

from scripts import train_prerank, train_rank, train_recall
from src import prerank
from src.runtime import write_json


class Interrupted(Exception):
    pass


def check(reference, device, out):
    cfg = json.loads((reference / "config.json").read_text())["config"]
    if not cfg["stage_limits"]:
        raise ValueError("Use a completed smoke run for this recovery test")
    records = {}
    with tempfile.TemporaryDirectory(dir=reference.parent) as directory:
        run = Path(directory)
        target = run / "base"
        target.mkdir()
        for name in ("requests.pkl", "meta.npy", "codes.npy", "catalog.npy", "split.json"):
            os.link(reference / name, run / name)
        for name in (
            "vectors.npy",
            "rank_train.exposures.npz",
            "rank_val.exposures.npz",
            "test.exposures.npz",
            "rank_val.candidates.npz",
            "test.candidates.npz",
        ):
            os.link(reference / "base" / name, target / name)

        for name, module in (
            ("tower", train_recall),
            ("lambdarank", train_prerank),
            ("mlp", train_rank),
            ("din", train_rank),
        ):
            args = SimpleNamespace(run=run, device=device, model=name, variant="base")
            container, attr = (prerank, "write_json") if name == "lambdarank" else (module, "save_torch")
            save = getattr(container, attr)

            def interrupt_after_checkpoint(path, content, save=save):
                save(path, content)
                if ".last." in Path(path).name:
                    raise Interrupted()

            random.seed(cfg["seed"])
            np.random.seed(cfg["seed"])
            torch.manual_seed(cfg["seed"])
            torch.set_num_threads(4)
            try:
                with patch.object(container, attr, interrupt_after_checkpoint):
                    module.train(args, cfg)
            except Interrupted:
                pass
            else:
                raise AssertionError("Trainer did not write a recovery checkpoint")
            module.train(args, cfg)
            if name == "lambdarank":
                with np.load(target / "rank_val.exposures.npz") as z:
                    features = z["x"]

                def predict(directory, features=features):
                    booster = lgb.Booster(model_file=str(directory / "lambdarank.best.txt"))
                    return booster.predict(features, num_threads=cfg["faiss_threads"])

                error = float(np.max(np.abs(predict(target) - predict(reference / "base"))))
                check_type = "maximum absolute validation score difference"
            else:

                def weights(directory, name=name):
                    return torch.load(directory / f"{name}.last.pt", map_location="cpu", weights_only=False)[
                        "model"
                    ]

                resumed, original = weights(target), weights(reference / "base")
                error = max(float(torch.max(torch.abs(v - original[k]))) for k, v in resumed.items())
                check_type = "maximum absolute final weight difference"
            if error > 1e-5:
                raise AssertionError(f"{name} resume mismatch {error}")
            records[name] = {
                "interrupted_after_checkpoint": 1,
                "comparison": check_type,
                "max_abs_error": error,
            }
    write_json(out, records)
    print(records, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check(args.run, args.device, args.out)

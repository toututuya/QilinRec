import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("fit", "tower_val", "rank_train", "rank_val", "test")
VARIANTS = ("base", "no_history", "no_content", "exposed_neg")


def configure(description, stage=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, default=os.environ.get("QILIN_DATA"))
    p.add_argument("--variant", choices=VARIANTS, default="base")
    p.add_argument("--device", default="cuda")
    if stage == "rank":
        p.add_argument("--model", choices=["mlp", "din"], required=True)
    args = p.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    cfg.update(yaml.safe_load(args.config.read_text()))
    args.run.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    manifest = args.run / "config.json"
    if manifest.exists() and json.loads(manifest.read_text())["sha256"] != fingerprint:
        raise ValueError("Run directory belongs to another configuration. Select a new --run path.")
    write_json(manifest, {"sha256": fingerprint, "config": cfg})
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --device cpu for an explicitly CPU-only run.")
    return args, cfg


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        torch.save(obj, f)
    Path(f.name).replace(path)


def evenly_spaced(n, cap):
    if cap <= 0 or cap >= n:
        return np.arange(n)
    return np.linspace(0, n - 1, cap, dtype=int)

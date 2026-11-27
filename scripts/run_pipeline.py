"""Run each stage in a separate process so ANN and training release their memory."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from src.runtime import ROOT, VARIANTS, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, default=os.environ.get("QILIN_DATA"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--ablations", action="store_true")
    p.add_argument(
        "--content-model-dir", type=Path, help="Also run content retrieval with this pinned BGE model"
    )
    p.add_argument("--content-out", type=Path, help="Content run directory (default: RUN/content)")
    p.add_argument("--content-config", type=Path, default=ROOT / "configs/content.yaml")
    p.add_argument(
        "--resume", action="store_true", help="Skip stages with completed outputs from this configuration"
    )
    args = p.parse_args()
    if args.data is None:
        p.error("Supply --data or QILIN_DATA")
    if args.content_out and not args.content_model_dir:
        p.error("--content-out requires --content-model-dir")
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    common = [
        "--config",
        str(args.config.resolve()),
        "--run",
        str(run),
        "--device",
        args.device,
        "--data",
        str(args.data.resolve()),
    ]
    sources = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("configs/*.yaml")])
    env = {
        "python": platform.python_version(),
        "system": platform.platform(),
        "device": args.device,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "packages": {
            k: importlib.metadata.version(k)
            for k in ["numpy", "torch", "faiss-cpu", "scipy", "lightgbm", "scikit-learn", "pyarrow", "PyYAML"]
        },
        "source_hashes": {
            f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources
        },
    }
    if (run / "config.json").exists():
        import yaml

        cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
        cfg.update(yaml.safe_load(args.config.read_text()))
        if cfg != json.loads((run / "config.json").read_text())["config"]:
            p.error("Existing run uses different settings")
    manifest = run / "environment.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())["source_hashes"]
        entrypoints = {"prepare", "train_recall", "build_index", "candidates", "train_prerank", "train_rank"}
        for name, digest in env["source_hashes"].items():
            # This optional module has its own run manifest. Legacy checkpoints
            # were trained before the content channel existed.
            if name == "src/recall/content.py":
                continue
            if (name.startswith("src/") or Path(name).stem in entrypoints) and previous.get(name) != digest:
                p.error(
                    f"Training implementation changed in {name}. Use a new run directory or restore the original source."
                )
    else:
        # The experiment report describes the environment at the start of training.
        # Replaying completed stages must preserve that original provenance.
        write_json(manifest, env)

    def step(module, product, *extra):
        if args.resume and (run / product).is_file():
            print("Complete", product, flush=True)
            return
        command = [sys.executable, "-m", "scripts." + module, *common, *extra]
        print("Running", module, *extra, flush=True)
        subprocess.run(command, check=True, cwd=ROOT)

    step("prepare", "split.json")
    for name in VARIANTS if args.ablations else ("base",):
        step("train_recall", f"{name}/tower_train.json", "--variant", name)
        step("build_index", f"{name}/faiss.json", "--variant", name)
        step("candidates", f"{name}/recall.json", "--variant", name)
        if name == "base":
            step("train_prerank", "base/prerank.json")
            for model in ("mlp", "din"):
                step("train_rank", f"base/{model}.json", "--model", model)
    if args.content_model_dir:
        import yaml

        dest = (args.content_out or run / "content").resolve()
        dest.mkdir(parents=True, exist_ok=True)
        if dest == run:
            p.error("Content artifacts need a separate output directory")
        config = yaml.safe_load(args.content_config.read_text())
        scope = ["encode_content.py", "run_content.py", "content_cascade.py", "align_content.py"]
        paths = [ROOT / "src/recall/content.py", args.content_config, *(ROOT / "scripts" / n for n in scope)]
        record = {
            "sources": {
                p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest() for p in paths
            },
            "base_config": json.loads((run / "config.json").read_text())["sha256"],
            "base_split": json.loads((run / "split.json").read_text())["stages"],
        }
        metadata = dest / "content_environment.json"
        if metadata.exists() and json.loads(metadata.read_text()) != record:
            p.error("Content implementation or inputs changed. Use a new --content-out directory.")
        write_json(metadata, record)
        content_args = ["--run", str(run), "--out", str(dest), "--device", args.device]

        def content_step(module, product, *extra):
            if args.resume and product is not None and (dest / product).is_file():
                return
            subprocess.run(
                [sys.executable, "-m", "scripts." + module, *content_args, *extra], check=True, cwd=ROOT
            )

        config_arg = ["--config", str(args.content_config.resolve())]
        content_step(
            "encode_content",
            "text_ready.json",
            *config_arg,
            "--data",
            str(args.data.resolve()),
            "--model-dir",
            str(args.content_model_dir.resolve()),
        )
        curve = dest / "training.json"
        finished = (
            curve.exists()
            and len(json.loads(curve.read_text())) == config["adapter_epochs"] + 1
            and (dest / "adapter.last.pt").is_file()
            and (dest / "adapter.best.pt").is_file()
        )
        content_step("run_content", "training.json" if finished else None, *config_arg, "--phase", "train")
        content_step("run_content", "policy.json", *config_arg, "--phase", "select")
        content_step("run_content", "test.json", *config_arg, "--phase", "test")
        content_step("content_cascade", "cascade.json")
        content_step("align_content", "alignment.json")
    print("Finished", run, flush=True)


if __name__ == "__main__":
    main()

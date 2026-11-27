"""Encode catalog text and pre-request clicked titles with a pinned Chinese BGE model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from filelock import FileLock
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from src.data import load_requests, valid_text
from src.runtime import ROOT, write_json

WEIGHTS = "354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026"
BLOBS = {
    "config.json": "f79ec8dfc5812a1d5d7d2e69253eb9054be2246d",
    "special_tokens_map.json": "a8b3208c2884c4efb86e49300fdd3dc877220cdf",
    "tokenizer.json": "cdb3043fc938fc918c06e66cf704c2ba58f88747",
    "tokenizer_config.json": "3a59388f0fd1bd22dec2ce7902c1be8e1fb84107",
    "vocab.txt": "ca4f9781030019ab9b253c6dcb8c7878b6dc87a5",
}


def verify_encoder(folder):
    with (folder / "model.safetensors").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != WEIGHTS:
            raise ValueError("Unexpected BGE weights")
    for file, expected in BLOBS.items():
        data = (folder / file).read_bytes()
        if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != expected:
            raise ValueError(f"Unexpected BGE input {file}")


def clean(value):
    return value.strip() if valid_text(value) else ""


@torch.inference_mode()
def encode(texts, model, tokenizer, limit, batch, device):
    vectors = np.zeros((len(texts), model.config.hidden_size), np.float16)
    for start in range(0, len(texts), batch):
        group = texts[start : start + batch]
        tokens = tokenizer(group, padding=True, truncation=True, max_length=limit, return_tensors="pt")
        features = model(**tokens.to(device)).last_hidden_state[:, 0]
        features = F.normalize(features.float(), dim=1)
        if not features.isfinite().all():
            raise ValueError("Nonfinite text embedding")
        vectors[start : start + len(group)] = features.half().cpu().numpy()
    vectors[[not s for s in texts]] = 0
    return vectors


def audit_manifest(run, root, out, cfg):
    split = json.loads((run / "split.json").read_text())
    revision = (root / "revision.txt").read_text(encoding="utf-8-sig").strip()
    if revision != split["revision"]:
        raise ValueError("Text and interaction data revisions differ")
    record = {
        "dataset_revision": revision,
        "encoder": cfg["model"],
        "encoder_revision": cfg["revision"],
        "weights_sha256": WEIGHTS,
        "dimension": cfg["dimension"],
        "article_tokens": cfg["article_tokens"],
        "profile_tokens": cfg["profile_tokens"],
        "recent_titles": cfg["recent_titles"],
        "catalog_items": split["catalog_total"],
        "catalog_sha256": hashlib.sha256((run / "catalog.npy").read_bytes()).hexdigest(),
        "request_hashes": {k: v["request_id_sha256"] for k, v in split["stages"].items()},
    }
    path = out / "text_manifest.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Embedding directory belongs to different inputs")
    write_json(path, record)
    return split


def prepare(args):
    cfg = yaml.safe_load(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    verify_encoder(args.model_dir)
    split = audit_manifest(args.run, args.data, args.out, cfg)
    rows = load_requests(args.run)
    ids = np.load(args.run / "catalog.npy")
    need = np.zeros(split["catalog_total"] + 1, bool)
    need[ids] = True
    need[np.concatenate([r.positives for r in rows["fit"]])] = True
    titles = {}
    for stage in rows.values():
        for r in stage:
            titles.update((int(i), "") for i in r.recent[: cfg["recent_titles"]])
    net = AutoModel.from_pretrained(args.model_dir, local_files_only=True, use_safetensors=True)
    if net.config.hidden_size != cfg["dimension"]:
        raise ValueError("Unexpected BGE hidden size")
    net = net.to(args.device).eval()
    if args.device.startswith("cuda"):
        net.half()
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    progress = args.out / "encoded_rows.json"
    done = json.loads(progress.read_text()) if progress.exists() else {}
    file = args.out / "articles.npy"
    if done and not file.is_file():
        raise ValueError("Missing catalog embedding file")
    matrix = np.lib.format.open_memmap(
        file, mode="r+" if file.exists() else "w+", dtype=np.float16, shape=(len(need), cfg["dimension"])
    )
    if matrix.shape != (len(need), cfg["dimension"]) or matrix.dtype != np.float16:
        raise ValueError("Invalid article matrix")
    shards = sorted(args.data.glob("notes-train-*-of-00005.parquet"))
    if len(shards) != 5:
        raise ValueError("Expected five notes shards")
    for shard in shards:
        pf = pq.ParquetFile(shard)
        cursor = 0
        for b in pf.iter_batches(batch_size=8192, columns=["note_idx", "note_title", "note_content"]):
            keys = b.column(0).to_numpy() + 1
            head = [clean(s) for s in b.column(1).to_pylist()]
            for i, s in zip(keys.tolist(), head, strict=True):
                if i in titles:
                    titles[i] = s
            cursor += len(b)
            if cursor <= done.get(shard.name, 0):
                continue
            keep = np.flatnonzero(need[keys])
            if len(keep):
                body = b.column(2).to_pylist()
                # Bound tokenizer work on long bodies before the token-level cap.
                texts = [" ".join(filter(None, (head[j], clean(body[j])[:512]))) for j in keep]
                matrix[keys[keep]] = encode(
                    texts, net, tok, cfg["article_tokens"], cfg["encoder_batch"], args.device
                )
            matrix.flush()
            done[shard.name] = cursor
            write_json(progress, done)
            if cursor % 65536 == 0 or cursor == pf.metadata.num_rows:
                print(shard.name, cursor, "/", pf.metadata.num_rows, flush=True)
    for stage, group in rows.items():
        dst = args.out / f"{stage}.profiles.npz"
        rids = np.array([r.rid for r in group])
        if dst.exists():
            with np.load(dst) as saved:
                np.testing.assert_array_equal(saved["request_ids"], rids)
            continue
        profiles = [" ".join(titles[i] for i in r.recent[: cfg["recent_titles"]] if titles[i]) for r in group]
        vec = encode(profiles, net, tok, cfg["profile_tokens"], cfg["encoder_batch"], args.device)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, vectors=vec, request_ids=rids)
        tmp.replace(dst)
        print(stage, len(rids), "profiles", sum(bool(s) for s in profiles), "with titles", flush=True)
    write_json(
        args.out / "text_ready.json",
        {
            "articles_requested": int(need.sum()),
            "catalog": len(ids),
            "shards": done,
            "model_revision": cfg["revision"],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "out", "data", "model-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/content.yaml")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with FileLock(args.out / ".content.lock", timeout=0):
        prepare(args)

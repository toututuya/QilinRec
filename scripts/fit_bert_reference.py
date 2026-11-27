"""Fit one shared BERT bi-encoder on pre-test, within-exposure contrasts."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from filelock import FileLock
from transformers import AutoModel, AutoTokenizer

from scripts.compare_text_baselines import BERT_REVISION, BERT_WEIGHTS, collect_text
from src.data import load_requests
from src.runtime import ROOT, save_torch, write_json


def tokenize(texts, tokenizer, limit):
    batches = []
    for start in range(0, len(texts), 512):
        x = tokenizer(
            texts[start : start + 512],
            padding="max_length",
            truncation=True,
            max_length=limit,
            return_tensors="np",
        )
        batches.append(x["input_ids"].astype(np.int32))
    return np.concatenate(batches)


def pool(model, ids, device, pad):
    input_ids = torch.as_tensor(ids, device=device, dtype=torch.long)
    mask = input_ids != pad
    hidden = model(input_ids=input_ids, attention_mask=mask).last_hidden_state.float()
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def train(args):
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["epochs"] != 1:
        raise ValueError("This reference is a single fixed pass")
    with (args.bert / "model.safetensors").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != BERT_WEIGHTS:
            raise ValueError("Unexpected base model")
    split = json.loads((args.run / "split.json").read_text())
    signature = {
        "config": cfg,
        "base_revision": BERT_REVISION,
        "base_sha256": BERT_WEIGHTS,
        "dataset_revision": split["revision"],
        "fit_requests_sha256": split["stages"]["fit"]["request_id_sha256"],
    }
    manifest = args.out / "design.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Use a separate folder for a different reference")
    write_json(manifest, signature)
    if (args.out / "training.json").exists():
        if not (args.out / "model.safetensors").is_file():
            raise ValueError("Missing completed model")
        return
    rows = load_requests(args.run)["fit"]
    print("Loaded fit requests", len(rows), flush=True)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    groups, fit = [], []
    for r in rows:
        negative = r.items[r.y == 0]
        if len(r.positives) and len(negative):
            take = rng.choice(negative, min(cfg["negatives"], len(negative)), replace=False)
            groups.append([int(rng.choice(r.positives)), *take.tolist()])
            fit.append(r)
    print("Sampled within-exposure groups", len(groups), flush=True)
    docs, histories = collect_text(args.data, groups, fit)
    print("Collected training texts", len(docs), flush=True)
    tokens = AutoTokenizer.from_pretrained(args.bert, local_files_only=True)
    keys = sorted(docs)
    lookup = {item: i + 1 for i, item in enumerate(keys)}
    item_tokens = tokenize([""] + [docs[i] for i in keys], tokens, cfg["note_tokens"])
    user_tokens = tokenize(histories, tokens, cfg["query_tokens"])
    print("Tokenized documents and histories", item_tokens.shape, user_tokens.shape, flush=True)
    picked = np.zeros((len(fit), cfg["negatives"] + 1), np.int32)
    for i, items in enumerate(groups):
        if histories[i] and docs[items[0]]:
            valid = [lookup[j] for j in items if docs[j]]
            if len(valid) > 1:
                picked[i, : len(valid)] = valid
    order = rng.permutation(np.flatnonzero(picked[:, 0]))
    model = (
        AutoModel.from_pretrained(args.bert, local_files_only=True, use_safetensors=True)
        .to(args.device)
        .train()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
    step_size = cfg["microbatch"] * cfg["accumulate"]
    steps = math.ceil(len(order) / step_size)
    begin, total_loss = 0, 0.0
    last = args.out / "progress.pt"
    if last.is_file():
        state = torch.load(last, map_location=args.device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        begin, total_loss = state["step"], state["loss"]
        del state
    amp = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    warmup = max(1, round(steps * cfg["warmup_fraction"]))
    for step in range(begin, steps):
        subset = order[step * step_size : (step + 1) * step_size]
        torch.manual_seed(cfg["seed"] + step)
        fraction = min((step + 1) / warmup, (steps - step) / max(1, steps - warmup))
        optimizer.param_groups[0]["lr"] = cfg["learning_rate"] * fraction
        optimizer.zero_grad(set_to_none=True)
        losses = 0.0
        for at in range(0, len(subset), cfg["microbatch"]):
            part = subset[at : at + cfg["microbatch"]]
            ix = picked[part]
            with torch.autocast(
                device_type=torch.device(args.device).type, dtype=torch.bfloat16, enabled=amp
            ):
                u = pool(model, user_tokens[part], args.device, tokens.pad_token_id)
                d = pool(model, item_tokens[ix.ravel()], args.device, tokens.pad_token_id).reshape(
                    len(part), ix.shape[1], -1
                )
                logits = (
                    (u[:, None] * d)
                    .sum(-1)
                    .masked_fill(torch.as_tensor(ix == 0, device=args.device), -torch.inf)
                )
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.zeros(len(part), device=args.device, dtype=torch.long), reduction="sum"
                )
            if not loss.isfinite():
                raise ValueError("Nonfinite BERT training loss")
            (loss / len(subset)).backward()
            losses += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += losses
        if (step + 1) % 100 == 0 or step + 1 == steps:
            print(step + 1, "/", steps, "loss", losses / len(subset), flush=True)
        if (step + 1) % 250 == 0 or step + 1 == steps:
            save_torch(
                last,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "loss": total_loss,
                },
            )
    model.save_pretrained(args.out, safe_serialization=True)
    tokens.save_pretrained(args.out)
    with (args.out / "model.safetensors").open("rb") as f:
        model_hash = hashlib.file_digest(f, "sha256").hexdigest()
    write_json(
        args.out / "training.json",
        {
            **signature,
            "fit_requests": len(rows),
            "eligible_two_label_requests": len(fit),
            "used_requests": len(order),
            "selected_exposed_negatives": int((picked[order, 1:] > 0).sum()),
            "steps": steps,
            "mean_training_ce": total_loss / len(order),
            "bf16": amp,
            "weights_sha256": model_hash,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "data", "bert", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--config", type=Path, default=ROOT / "configs/bert_reference.yaml")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.out / ".fit.lock"), timeout=0):
        train(args)

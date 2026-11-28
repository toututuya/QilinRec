"""Score BM25 and a frozen Chinese BERT bi-encoder on the exact QilinRec exposure cohorts."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import jieba
import numpy as np
import pyarrow.parquet as pq
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from scripts.compare_qilin import COMMIT, author_metric, official_queries
from scripts.encode_content import clean
from scripts.verify_run import request_ndcg
from src.data import load_requests
from src.metrics import retrieval_metrics
from src.runtime import write_json

BERT_REVISION = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"
BERT_WEIGHTS = "3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3"


def collect_text(root, pools, rows):
    # Match the V2 BGE text construction before model-specific tokenization.
    needed = {i for pool in pools for i in pool}
    recent = {i for row in rows for i in row.recent[:10]}
    documents, titles = {}, {}
    shards = sorted(root.glob("notes-train-*-of-00005.parquet"))
    if len(shards) != 5:
        raise ValueError("Expected all five note shards")
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(
            batch_size=4096, columns=["note_idx", "note_title", "note_content"]
        ):
            keys = batch.column(0).to_numpy() + 1
            for at, item in enumerate(keys.tolist()):
                if item not in needed and item not in recent:
                    continue
                title = clean(batch.column(1)[at].as_py())
                if item in recent:
                    titles[item] = title
                if item in needed:
                    body = clean(batch.column(2)[at].as_py())[:512]
                    documents[item] = " ".join(filter(None, (title, body)))
    if set(documents) != needed or set(titles) != recent:
        raise ValueError("Unresolved catalog IDs")
    histories = [" ".join(titles[i] for i in r.recent[:10] if titles[i]) for r in rows]
    return documents, histories


def bm25_rankings(pools, histories, documents):
    def tokenize(text):
        return [t for t in jieba.lcut(text) if t.strip()]

    corpus = {i: tokenize(t) for i, t in documents.items()}
    ranked, tied = [], []
    for items, history in zip(pools, histories, strict=True):
        tokens = [corpus[i] for i in items]
        query = tokenize(history)
        scores = (
            BM25Okapi(tokens, k1=1.5, b=0.75, epsilon=0.25).get_scores(query)
            if query and any(tokens)
            else np.zeros(len(items))
        )
        ids = np.asarray(items)
        ranked.append(ids[np.lexsort((ids, -scores))].tolist())
        tied.append(bool((scores == scores.max()).sum() > 1))
    return ranked, tied


@torch.inference_mode()
def encode(texts, model, tokenizer, limit, device):
    # Compute the released DPR masked mean in FP32, without cosine normalization.
    output = np.zeros((len(texts), model.config.hidden_size), np.float32)
    positions = sorted((j for j, t in enumerate(texts) if t), key=lambda j: len(texts[j]))
    for start in range(0, len(positions), 48):
        part = positions[start : start + 48]
        inputs = tokenizer(
            [texts[j] for j in part], padding=True, truncation=True, max_length=limit, return_tensors="pt"
        ).to(device)
        hidden = model(**inputs).last_hidden_state.float()
        mask = inputs.attention_mask.unsqueeze(-1)
        output[part] = ((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy()
    return output


def bert_rankings(pools, histories, documents, model_dir, device, expected_weights=BERT_WEIGHTS):
    with (model_dir / "model.safetensors").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected_weights:
            raise ValueError("Unexpected BERT weights")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModel.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
    )
    if device.startswith("cuda"):
        model.half()
    item_ids = sorted(documents)
    notes = encode([documents[i] for i in item_ids], model, tokenizer, 64, device)
    users = encode(histories, model, tokenizer, 128, device)
    lookup = {item: j for j, item in enumerate(item_ids)}
    ranked = []
    for ids, user in zip(pools, users, strict=True):
        ix = [lookup[i] for i in ids]
        score = (notes[ix] * user).sum(axis=1)
        if user.any():
            score[~notes[ix].any(axis=1)] = -np.inf
        ids = np.asarray(ids)
        ranked.append(ids[np.lexsort((ids, -score))].tolist())
    return ranked


def run(args):
    rows = load_requests(args.run)["test"]
    indices, pools, qrels = official_queries(args.data, args.qrels, rows)
    rows = [rows[i] for i in indices]
    documents, histories = collect_text(args.data, pools, rows)
    print("Text ready", len(rows), len(documents), flush=True)
    bm, ties = bm25_rankings(pools, histories, documents)
    print("BM25 ready", flush=True)
    bert = bert_rankings(pools, histories, documents, args.bert, args.device)
    rankings = {"stored_order": pools, "bm25": bm, "bert_base_frozen": bert}
    scorer = author_metric(args.scorer)
    controls = json.loads(args.controls.read_text())
    out = {
        "official_revision": COMMIT,
        "bert_revision": BERT_REVISION,
        "bert_weights_sha256": BERT_WEIGHTS,
        "bert_pooling": "masked mean then dot product; frozen public MLM backbone",
        "input": {
            "recent_titles": 10,
            "body_characters": 512,
            "bert_query_tokens": 128,
            "bert_note_tokens": 64,
        },
        "bm25": {
            "scope": "per official exposure pool",
            "tokenizer": "jieba default accurate mode",
            "k1": 1.5,
            "b": 0.75,
            "epsilon": 0.25,
        },
        "packages": {
            k: importlib.metadata.version(k) for k in ("jieba", "rank-bm25", "torch", "transformers")
        },
        "unique_exposed_notes": len(documents),
    }
    if args.trained_bert is not None:
        training = json.loads((args.trained_bert / "training.json").read_text())
        split = json.loads((args.run / "split.json").read_text())
        if (
            training["base_sha256"] != BERT_WEIGHTS
            or training["fit_requests_sha256"] != split["stages"]["fit"]["request_id_sha256"]
        ):
            raise ValueError("The supervised reference used a different base model or fit window")
        rankings["bert_fit_one_epoch"] = bert_rankings(
            pools, histories, documents, args.trained_bert, args.device, training["weights_sha256"]
        )
        out["supervised_reference"] = training
    for name, n in (("config_first_1000", 1000), ("all_requests", len(rows))):
        truth = {j: qrels[j] for j in range(n)}
        part = {
            "requests": n,
            "pairs": sum(map(len, pools[:n])),
            "qrels_rows": sum(map(len, truth.values())),
            "empty_profile": sum(not t for t in histories[:n]),
            "bm25_tied_best": sum(ties[:n]),
            "results": {},
        }
        for key, ranks in rankings.items():
            group = ranks[:n]
            for candidates, ranking in zip(pools[:n], group, strict=True):
                if len(candidates) != len(ranking) or set(candidates) != set(ranking):
                    raise ValueError("Lost official candidate")
            author = scorer(dict(enumerate(group)), truth, (10, 100))
            own = retrieval_metrics(group, rows[:n], controls["catalog_size"], (10,))
            ndcg = float(request_ndcg(rows[:n], group).mean())
            np.testing.assert_allclose(ndcg, own["ndcg@10"], atol=1e-12)
            np.testing.assert_allclose(own["mrr@10"], author["MRR@10"], atol=1e-12)
            part["results"][key] = {**author, "NDCG@10": ndcg}
        prior = controls[name]
        assert (part["pairs"], part["qrels_rows"], part["empty_profile"]) == (
            prior["exposed_pairs"],
            prior["qrels_rows"],
            prior["empty_text_profiles"],
        )
        for key in ("MRR@10", "MAP@10"):
            np.testing.assert_allclose(
                part["results"]["stored_order"][key],
                prior["results"]["stored_order"]["official"][key],
                atol=1e-12,
            )
        out[name] = part
        print(
            name,
            {k: (v["MRR@10"], v["MAP@10"], v["NDCG@10"]) for k, v in part["results"].items()},
            flush=True,
        )
    write_json(args.out, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "data", "qrels", "scorer", "bert", "controls", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--trained-bert", type=Path, help="Also evaluate a completed one-epoch exposure-supervised reference"
    )
    run(p.parse_args())

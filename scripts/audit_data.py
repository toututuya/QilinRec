"""Audit the two recommendation tables and the complete note catalog, in batches."""

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REVISION = "2d79000cef450b99f6423357ce86e5c132d32dd5"


def missing(array):
    count = array.null_count
    if pa.types.is_floating(array.type):
        mask = pc.invert(pc.is_finite(array))
    elif pa.types.is_string(array.type):
        mask = pc.is_in(
            pc.utf8_lower(pc.utf8_trim_whitespace(array)), value_set=pa.array(["", "nan", "none", "null"])
        )
    else:
        return count
    return count + int(pc.sum(pc.fill_null(mask, False)).as_py() or 0)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_requests(path, split):
    starts, ends, sizes, lengths = [], [], [], []
    request_ids, users, exposed, history = set(), set(), set(), set()
    counts, absent = Counter(), Counter()
    labels = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        for name, arr in zip(batch.schema.names, batch.columns):
            absent[name] += missing(arr)
        details = batch.column("rec_result_details_with_idx").values
        for key in details.type.names:
            absent["details." + key] += missing(details.field(key))
        for rec in batch.select(
            ["request_idx", "user_idx", "recent_clicked_note_idxs", "rec_result_details_with_idx"]
        ).to_pylist():
            rid, uid = rec["request_idx"], rec["user_idx"]
            if rid in request_ids or uid is None or rid is None:
                raise ValueError(f"Duplicate/missing request or user ID in {split}")
            request_ids.add(rid)
            users.add(uid)
            events = rec["rec_result_details_with_idx"]
            recent = rec["recent_clicked_note_idxs"] or []
            if not events:
                raise ValueError("Empty request has no reliable time")
            by_id = {}
            for event in events:
                key, label = event["note_idx"], event["click"]
                if label not in (0, 1):
                    raise ValueError("Unexpected click label")
                labels[label] += 1
                by_id[key] = max(label, by_id.get(key, 0))
            stamps = [e["request_timestamp"] for e in events]
            starts.append(min(stamps))
            ends.append(max(stamps))
            sizes.append(len(by_id))
            lengths.append(len(recent))
            exposed.update(by_id)
            history.update(recent)
            npos = sum(by_id.values())
            counts["unique_pairs"] += len(by_id)
            counts["unique_positives"] += npos
            counts["mixed_groups"] += 0 < npos < len(by_id)
            counts["no_click_groups"] += npos == 0
            counts["history_overlap_positives"] += len({i for i, y in by_id.items() if y} & set(recent))
    result = dict(counts)
    result.update(
        requests=len(request_ids),
        users=len(users),
        exposed_items=len(exposed),
        history_items=len(history),
        impressions=sum(labels.values()),
        raw_clicks=labels[1],
        raw_nonclicks=labels[0],
        missing=dict(absent),
        request_width_min_median_max=np.quantile(sizes, [0, 0.5, 1]).tolist(),
        history_length_min_median_max=np.quantile(lengths, [0, 0.5, 1]).tolist(),
        start_min=min(starts),
        start_max=max(starts),
        end_max=max(ends),
        schema={f.name: str(f.type) for f in pq.ParquetFile(path).schema_arrow},
    )
    return result, request_ids, users, exposed, history, np.array(ends)


def audit(root):
    if (root / "revision.txt").read_text(encoding="utf-8-sig").strip() != REVISION:
        raise ValueError("Unexpected Qilin revision")
    notes = sorted(root.glob("notes-train-*-of-00005.parquet"))
    if len(notes) != 5:
        raise ValueError("Expected the five note shards")
    catalog, absent, shards = set(), Counter(), {}
    for path in notes:
        pf = pq.ParquetFile(path)
        shards[path.name] = pf.metadata.num_rows
        for b in pf.iter_batches(batch_size=16384):
            catalog.update(b.column("note_idx").to_pylist())
            for name, arr in zip(b.schema.names, b.columns):
                absent[name] += missing(arr)
    if len(catalog) != sum(shards.values()) or None in catalog:
        raise ValueError("Invalid catalog primary key")
    rec_paths = [root / f"recommendation_{s}.parquet" for s in ("train", "test")]
    tr, te = [inspect_requests(path, path.stem) for path in rec_paths]
    return {
        "revision": REVISION,
        "hashes": {p.name: digest(p) for p in notes + rec_paths},
        "catalog_items": len(catalog),
        "note_rows": shards,
        "notes_missing": dict(absent),
        "notes_schema": {f.name: str(f.type) for f in pq.ParquetFile(notes[0]).schema_arrow},
        "train": tr[0],
        "test": te[0],
        "users_total": len(tr[2] | te[2]),
        "exposed_items_total": len(tr[3] | te[3]),
        "shared_request_ids": len(tr[1] & te[1]),
        "exposures_outside_catalog": len((tr[3] | te[3]) - catalog),
        "histories_outside_catalog": len((tr[4] | te[4]) - catalog),
        "train_requests_purged_at_test": int(np.sum(tr[5] >= te[0]["start_min"])),
    }


def report(a):
    utc = lambda n: datetime.fromtimestamp(n, UTC).isoformat()
    t, e = a["train"], a["test"]
    lines = [
        "# Qilin 推荐数据审计",
        "",
        (
            f"数据版本 `{REVISION}`。共 {a['users_total']:,} 名用户，"
            f"{t['requests'] + e['requests']:,} 个请求，曝光涉及 {a['exposed_items_total']:,} 个物品。"
            f"五个笔记分片合计 {a['catalog_items']:,} 个唯一物品。"
        ),
        "",
        "| 官方分片 | 请求 | 用户 | 原始曝光 | 原始点击 | 唯一请求物品对 | 唯一正例 | 混合标签请求 | 零点击请求 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in [("train", t), ("test", e)]:
        lines.append(
            "| "
            + " | ".join(
                [name]
                + [
                    f"{s[k]:,}"
                    for k in (
                        "requests",
                        "users",
                        "impressions",
                        "raw_clicks",
                        "unique_pairs",
                        "unique_positives",
                        "mixed_groups",
                        "no_click_groups",
                    )
                ]
            )
            + " |"
        )
    lines.extend(["", "## 时间与标签", ""])
    for name, s in [("train", t), ("test", e)]:
        lines.append(
            f"- {name} 请求开始时间为 {utc(s['start_min'])} 至 {utc(s['start_max'])}，最后曝光为 {utc(s['end_max'])}。"
            f"每组去重候选数的最小值、中位数、最大值为 {s['request_width_min_median_max']}；"
            f"官方近期点击序列长度为 {s['history_length_min_median_max']}。"
        )
    lines.extend(
        [
            "",
            (
                f"官方只提供推荐训练和测试分片。共有 {a['train_requests_purged_at_test']:,} 个训练请求在最早测试请求开始时尚未结束或尚未开始。"
                "建模前先剔除这些请求，再在训练期内按时间建立验证窗口，跨界请求整组剔除。"
            ),
            "",
            (
                "request_idx 标识请求，user_idx 标识用户。每个请求的 rec_result_details_with_idx 包含真实曝光的物品、位置、曝光时间、点击、点赞、收藏、评论、分享和停留反馈。"
                "同一请求重复曝光的同一物品按最大 click 合并，可构建有真实正负反馈的 LambdaRank 分组。"
                "recent_clicked_note_idxs 提供用户点击历史，缺少每次历史点击的时间；审慎的历史重建见模型设计。"
            ),
            "",
            (
                f"两个分片共有 {a['shared_request_ids']} 个重叠请求 ID。曝光中有 {a['exposures_outside_catalog']} 个不同 ID、"
                f"历史中有 {a['histories_outside_catalog']} 个不同 ID 未进入笔记目录。目录提供标题、正文、三级类目、笔记类型等字段。"
                "该版本支持发布目录上的全库召回评测。目录没有发布时间或历史快照，不能还原当时在线可投放库存。"
                "缺少用户属性表；固定目录内容和有时间依据的交互可以支持双塔训练。测试特有的 rec_results、query、原始位置、事后动作、全期统计列均不作为模型输入。"
            ),
            "",
            "## 缺失与字段类型",
            "",
            (
                "缺失数计入 Arrow null、非有限浮点数、空字符串及文本占位符 nan/none/null。列表本身为空与 null 分别处理。"
                "全部字段的类型、逐列缺失数、原始文件 SHA-256 和分片行数见 [机器可读审计](../results/audit.json)。"
            ),
            "",
            "| 表 | 存在缺失的字段及行数 |",
            "|---|---|",
        ]
    )
    for name, m in [("notes", a["notes_missing"]), ("train", t["missing"]), ("test", e["missing"])]:
        lines.append(
            "| " + name + " | " + (", ".join(f"{k} ({v:,})" for k, v in m.items() if v) or "无") + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=os.environ.get("QILIN_DATA"), required="QILIN_DATA" not in os.environ
    )
    parser.add_argument("--json", type=Path, default=Path("results/audit.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/data_audit.md"))
    args = parser.parse_args()
    result = audit(args.data)
    for path, content in [
        (args.json, json.dumps(result, indent=2, ensure_ascii=False)),
        (args.report, report(result)),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + ("" if content.endswith("\n") else "\n"), encoding="utf-8")
    print(
        "Audited",
        result["catalog_items"],
        "catalog items and",
        result["train"]["requests"] + result["test"]["requests"],
        "requests.",
    )

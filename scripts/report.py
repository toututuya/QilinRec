"""Export aggregate evidence and tables from completed, verified training runs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.runtime import STAGES, VARIANTS, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def table(columns, rows):
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
            *["| " + " | ".join(str(v) for v in row) + " |" for row in rows],
        ]
    )


def fmt(v):
    return "—" if v is None else f"{v:.6f}"


def export(run, out):
    base = run / "base"
    split, env, config, verify = [
        read(run / name) for name in ("split.json", "environment.json", "config.json", "verification.json")
    ]
    recall = read(base / "recall.json")
    pre = read(base / "prerank.json")
    rank = {name: read(base / f"{name}.json") for name in ["mlp", "din"]}
    variants = [v for v in VARIANTS if (run / v / "recall.json").is_file()]
    towers = {v: read(run / v / "tower_train.json") for v in variants}
    ann = {v: read(run / v / "faiss.json") for v in variants}
    alternative = {v: read(run / v / "recall.json")["test"]["tower"] for v in variants if v != "base"}
    exp, cascade = {}, {}
    records = []

    def record(scope, model, stage, metrics):
        records.append(dict(scope=scope, model=model, split=stage, **metrics))

    for stage, met in recall.items():
        for method, m in met.items():
            record("observed_catalog" if stage == "test_observed_catalog" else "catalog", method, stage, m)
    for variant, m in alternative.items():
        record("catalog_tower_ablation", variant, "test", m)
    for stage in ["rank_val", "test"]:
        exp[stage] = {name: pre[stage][name]["exposure"] for name in ("rrf", "lambdarank")}
        cascade[stage] = {name + "_top50": pre[stage][name]["cascade"] for name in ("rrf", "lambdarank")}
        for name, rr in rank.items():
            exp[stage][name] = rr[stage]["exposure"]
            cascade[stage][name + "_top10"] = rr[stage]["cascade"]
        for name, m in exp[stage].items():
            record("exposures", name, stage, m)
        for name, m in cascade[stage].items():
            record("cascade", name, stage, m)
    for v, bench in ann.items():
        for method in ("flat", "ivf"):
            record("faiss_benchmark", v + "_" + method, "test_subset", bench[method])
    out.mkdir(parents=True, exist_ok=True)
    cols = ["scope", "model", "split", "requests", "click_requests", "positives"]
    cols += sorted(set().union(*(r.keys() for r in records)) - set(cols))
    with (out / "experiments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(records)
    artifacts = {}
    for v in variants:
        for name in ["tower.best.pt", "tower.last.pt", "vectors.npy", "catalog.ivf.index"]:
            p = run / v / name
            artifacts[str(p.relative_to(run))] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    for name in [
        "itemcf.npz",
        "lambdarank.best.txt",
        "lambdarank.last.json",
        "mlp.best.pt",
        "mlp.last.pt",
        "din.best.pt",
        "din.last.pt",
        "rank_feature_scale.npy",
        "test.candidates.npz",
        "test.shortlist.npz",
        "test.mlp.top.npz",
        "test.din.top.npz",
    ]:
        p = base / name
        artifacts[str(p.relative_to(run))] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    evidence = {
        "config": config,
        "environment": env,
        "split": split,
        "verification": verify,
        "artifacts": artifacts,
        "tower_training": towers,
        "index_benchmark": ann,
        "prerank_training": read(base / "lgb_train.json"),
        "ctr_training": {
            n: {"best_epoch": r["best_epoch"], "history": r["history"]} for n, r in rank.items()
        },
    }
    write_json(out / "evidence.json", evidence)
    test = recall["test"]
    n = split["catalog"]
    title = "完整目录正式实验" if n == split["catalog_total"] else "小规模端到端验证"
    report = [
        f"# QilinRec {title}",
        "",
        (
            f"设备 {env['gpu'] or env['device']}，PyTorch {env['packages']['torch']}，随机种子 {config['config']['seed']}。"
            f"本次索引包含 {n:,} 个物品。测试覆盖 {test['rrf']['requests']:,} 个请求、{test['rrf']['positives']:,} 个去重点击正例。"
            "以下均由保存的模型及预测生成。"
        ),
        "",
        table(
            ["窗口", "请求", "曝光物品对", "点击正例", "有可验证历史的请求"],
            [
                [
                    s,
                    *[
                        f"{split['stages'][s][k]:,}"
                        for k in ("requests", "exposures", "positives", "with_history")
                    ],
                ]
                for s in STAGES
            ],
        ),
        "",
        "## A　目录召回",
        "",
        table(
            [
                "通道",
                "Recall@10",
                "Recall@50",
                "Recall@100",
                "HitRate@100",
                "NDCG@100",
                "Coverage@100",
                "命中正例数",
            ],
            [
                [
                    name,
                    *[
                        fmt(m[s])
                        for s in [
                            "recall@10",
                            "recall@50",
                            "recall@100",
                            "hr@100",
                            "ndcg@100",
                            "coverage@100",
                        ]
                    ],
                    m["hits@100"],
                ]
                for name, m in test.items()
            ],
        ),
        "",
        "召回指标以完整请求真值为分母。各截止位的 HitRate、NDCG、覆盖率和微平均留存率均在 [CSV](experiments.csv) 中。",
        "",
    ]
    observed = recall["test_observed_catalog"]["rrf_exact_tower"]
    diagnostic = verify["positive_diagnostics"]
    interval = verify["din_minus_mlp"]
    report += [
        (
            f"全部正例中，有 {diagnostic['not_in_fit_clicks']:,} 个对应物品在召回拟合阶段没有记录过点击，"
            f"{diagnostic['not_in_fit_observations']:,} 个在该阶段的曝光与近期历史中均未出现。"
            f"{diagnostic['excluded_seen']:,} 个被已交互过滤，{diagnostic['outside_catalog']:,} 个在本次目录之外。"
            "各计数可能重叠，均保留在分母中。"
        ),
        "",
    ]
    report += [
        (
            f"在召回拟合期已出现的目录物品上另建 FlatIP，RRF 的 Recall@100 为 {fmt(observed['recall@100'])}，"
            f"找回 {observed['hits@100']:,} 个测试正例。该实验同时改变了可见范围和双塔检索方式，不能将差异全部归因于目录大小。"
        ),
        "",
        "## B / C　曝光列表上的预排序与精排",
        "",
        table(
            ["模型", "AUC", "LogLoss", "用户 GAUC", "有效用户", "NDCG@10", "MRR@10", "MAP@10"],
            [
                [
                    name,
                    fmt(m["auc"]),
                    fmt(m.get("logloss")),
                    fmt(m["gauc"]),
                    m["gauc_users"],
                    fmt(m["ndcg@10"]),
                    fmt(m["mrr@10"]),
                    fmt(m["map@10"]),
                ]
                for name, m in exp["test"].items()
            ],
        ),
        "",
        (
            "每个模型打分同一份完整曝光列表。RRF 在未进入召回的曝光上得 0 分。并列按 note ID 升序。"
            "精排 LogLoss 在全部曝光物品对上计算，GAUC 按用户的曝光物品对数加权。"
        ),
        "",
        "## 逐级保留的真实点击",
        "",
        table(
            ["阶段", "截止位", "保留正例 / 全部正例", "微平均留存率", "请求宏平均 Recall", "NDCG@10"],
            [
                [
                    name,
                    k,
                    f"{m[f'hits@{k}']:,} / {m['positives']:,}",
                    fmt(m[f"retained@{k}"]),
                    fmt(m[f"recall@{k}"]),
                    fmt(m["ndcg@10"]),
                ]
                for name, k, m in [
                    ("RRF", 100, test["rrf"]),
                    ("RRF 截断对照", 50, cascade["test"]["rrf_top50"]),
                    ("LambdaRank", 50, cascade["test"]["lambdarank_top50"]),
                    ("MLP", 10, cascade["test"]["mlp_top10"]),
                    ("DIN", 10, cascade["test"]["din_top10"]),
                ]
            ],
        ),
        "",
        "MLP 和 DIN 共享 LambdaRank 的 Top 50 输入。此表使用全链路预测，不要求真实点击已进入候选。",
        "",
        "同一用户的请求相关。以用户为单位做 2,000 次配对自助抽样，计算 DIN 减 MLP 的请求宏平均 NDCG@10 差值。",
        "",
        table(
            ["口径", "差值", "95% 百分位区间"],
            [
                [label, fmt(m["delta_ndcg10"]), f"[{fmt(m['confidence95'][0])}, {fmt(m['confidence95'][1])}]"]
                for label, m in [("曝光", interval["exposure"]), ("全链路", interval["cascade"])]
            ],
        ),
        "",
        "## D　消融",
        "",
        table(
            ["比较", "Recall@100", "差值（绝对）", "命中数"],
            [
                [name, fmt(m["recall@100"]), fmt(m["recall@100"] - ref["recall@100"]), m["hits@100"]]
                for name, m, ref in [
                    ("RRF 去掉 ItemCF", test["no_itemcf"], test["rrf"]),
                    ("RRF 去掉双塔", test["no_tower"], test["rrf"]),
                ]
                + [(v, m, test["tower"]) for v, m in alternative.items()]
            ],
        ),
        "",
        (
            "通道消融仅改变融合输入。双塔消融各自重新训练、独立选择验证最佳轮次并重建索引，目录、划分和种子一致。"
            "no_history 关闭用户塔的历史输入，no_content 关闭类目与标题输入。exposed_neg 在过滤后的批内对比项之外每行最多补一个真实曝光未点击样本。"
        ),
        "",
        "## FAISS 与训练记录",
        "",
    ]
    a = ann["base"]
    report += [
        table(
            ["索引", "构建秒数", "批量摊销毫秒/请求", "同一子集 Recall@100"],
            [
                [
                    name,
                    f"{a[name + '_build_s']:.3f}",
                    f"{a[name]['batch_amortized_ms']:.3f}",
                    fmt(a[name]["recall@100"]),
                ]
                for name in ("flat", "ivf")
            ],
        ),
        "",
        (
            f"向量编码 {a['encode_s']:.3f} 秒。比较了 {a['flat']['queries']} 个等距抽取的测试请求，批大小 128、CPU 线程 {a['threads']}。"
            f"IVF 的 nlist={a['nlist']}、nprobe={a['nprobe']}，Top 100 与精确近邻的平均集合重合率 {fmt(a['ivf_flat_overlap100'])}。"
            "查询计时包含过滤和结果整理，不含用户塔推理。相同向量的分数并列会影响集合重合率。"
        ),
        "",
        (
            f"主双塔选中第 {a['best_epoch']} 轮，LambdaRank 选中第 {evidence['prerank_training']['best_iteration']} 棵树，"
            f"MLP 和 DIN 分别选中第 {rank['mlp']['best_epoch']} 和 {rank['din']['best_epoch']} 轮。"
            "每轮 Loss/验证指标、依赖版本、划分指纹、检查点与索引 SHA-256 见 [实验证据](evidence.json)。"
        ),
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Wrote", out, "with", len(records), "metric records", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    export(args.run, args.out)

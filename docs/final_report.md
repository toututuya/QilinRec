# QilinRec 最终实验报告

在 1,983,938 篇笔记中，精确检索的 BGE + Adapter + 热度为 11,115 个测试请求找回 **786** 个点击，宏平均 Recall@100 为 **2.5338%**。使用同一评分函数的 FAISS IVFFlat 找回 **734** 个点击，在同一 CPU 上将单请求检索 P95 从 **143.43 ms** 降至 **11.99 ms**。精确检索作为默认质量基准，IVF 作为可选加速路径。下列实测值均来自 Atlas。[39 行实验表](../results/final/experiments.csv) · [指标、配置、请求与权重指纹](../results/final/evidence.json)

## 同一曝光候选上的文本排序

按测试 Parquet 原始顺序取公开 DPR 配置指定的前 1,000 个请求。将 `rec_results` 与逐请求真实曝光集合核对，将官方 qrels 与点击标签核对。这个子集有 18,437 个曝光对、5,382 个点击对。全部 11,115 个请求有 179,401 个曝光对、44,683 个唯一点击对，候选数中位数为 9，范围为 1–79。MRR 和 MAP 使用作者评分函数与原始 qrels，完整 qrels 含两条重复点击。NDCG 使用唯一点击的二值增益并按请求取平均。[作者配置](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/config/recommendation_dpr_config.yaml#L49-L59) · [作者评分函数](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/evaluator.py#L16-L90)

| Atlas 实测模型 | 前 1,000 MRR@10 | MAP@10 | NDCG@10 | 全部 11,115 MRR@10 | MAP@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.5600 | 0.4092 | 0.5527 | 0.5531 | 0.4347 | 0.5671 |
| BERT 双编码器，冻结预训练权重 | 0.5505 | 0.4073 | 0.5454 | 0.5352 | 0.4234 | 0.5551 |
| BERT 双编码器，拟合期训练一轮 | 0.5823 | 0.4266 | 0.5718 | 0.5708 | 0.4526 | 0.5844 |
| BGE，冻结 | 0.5942 | 0.4296 | 0.5703 | 0.5735 | 0.4527 | 0.5827 |
| BGE + 热度 | 0.5974 | 0.4333 | 0.5732 | 0.5769 | 0.4550 | 0.5847 |
| BGE + Adapter | 0.5911 | 0.4397 | 0.5770 | 0.5764 | 0.4568 | 0.5860 |
| BGE + Adapter + 热度 | 0.5948 | 0.4398 | 0.5778 | 0.5784 | 0.4574 | 0.5869 |
| V1 DIN，单独排序 | 0.5958 | 0.4433 | 0.5806 | 0.5815 | 0.4626 | 0.5914 |
| V2 召回分数与两级排序融合 | 0.5882 | 0.4404 | 0.5781 | 0.5819 | 0.4623 | 0.5918 |

- **输入与打分。** 文本对照共用请求前最近 10 篇点击笔记的标题，以及候选的标题和前 512 个正文字符。BM25 采用 jieba 精确分词，各请求在自己的候选池内计算 IDF，Okapi 参数为 `k1=1.5`、`b=0.75`、`epsilon=0.25`。BERT 在 query 和笔记两侧共享编码器，对有效 token 做均值池化并取内积，两侧长度上限分别为 128、64。文本基线同分按笔记 ID 升序。作者代码中的候选集合迭代与并列顺序可能不同。[作者 BM25 实现](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/evaluator.py#L96-L141) · [作者双编码器](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/model_factory.py#L37-L93) · [本项目文本评测](../scripts/compare_text_baselines.py)
- **训练对照。** 两份 BERT 由同一份公开中文 BERT 权重初始化。一份全程冻结，另一份只用召回拟合期，从同一请求中采一个真实点击和至多三个曝光未点，用组内交叉熵训练共享编码器。35,424 个拟合请求中，35,231 个同时有两类标签，35,085 个有可用训练文本，共采到 99,505 个曝光未点。RTX 4090 上以 BF16、AdamW、学习率 `2e-5` 和 5% warmup 训练一轮，共 1,462 步。轮数在评分前固定。BGE 各组复用此前的冻结向量、第 1 轮 Adapter 和验证选定的热度系数 0.03。[中文 BERT 固定版本](https://huggingface.co/google-bert/bert-base-chinese/tree/8f23c25b06e129b6c986331a13d8d025a92cf0ea) · [本项目训练配置](../configs/bert_reference.yaml) · [BGE 训练与选择](content_retrieval.md)
- **如何读论文数字。** Qilin 表 10 的推荐任务中，BM25 的 MRR@10 / MAP@10 为 **0.5379 / 0.3933**，BERT 双编码器为 **0.6067 / 0.4548**，交叉编码器为 **0.6346 / 0.4786**，DCN-V2 为 **0.6307 / 0.4651**，VLM 为 **0.6394 / 0.4890**。表 10 未列 NDCG 或逐请求清单。本次核对的作者仓库与发布页没有该表微调后的 BERT 权重。作者 DPR 公开配置的长度、负例数和优化器也与本项目不同。可以复核同一批请求上本项目模型之间的差值，表 10 的数值仅作外部参照。[SIGIR 2025 论文 §5.1、表 10](https://xuanyuan14.github.io/files/SIGIR25Chen.pdf#page=9) · [作者训练配置](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/config/recommendation_dpr_config.yaml) · [其余截断位置](qilin_comparison.md)
- **结果解释。** 前 1,000 个请求上，训练 BERT 将 MRR 从 0.5505 提至 0.5823。BGE + Adapter + 热度相对这份训练对照提高 0.0125 MRR、0.0132 MAP 和 0.0060 NDCG。各组件的效果取决于指标。BGE + 热度的 MRR 最高，而在这批曝光候选上单独 DIN 的 MAP、NDCG 最高。完整测试集的候选分布与前 1,000 个请求不同，两组结果分别报告。[同池消融](../results/content/qilin_comparison.json)

## 全库召回、拟合期可见性与最终排序

下表每行按原召回分数截断，44,683 个唯一点击对全部进入分母。Recall 为先算每个请求找回的点击比例，再对 11,115 个请求取平均。命中数是成功找回的请求–笔记对总数。

| 全库检索方法 | Recall@10 | Recall@50 | Recall@100 | Top 100 命中 |
|---|---:|---:|---:|---:|
| V1 热门 | 0.2072% | 0.3641% | 0.4800% | 272 |
| V1 ItemCF | 0.0029% | 0.0194% | 0.0390% | 21 |
| V1 ID 双塔 · IVF | 0.0000% | 0.0145% | 0.0251% | 8 |
| V1 三路 RRF | 0.1692% | 0.2713% | 0.3504% | 174 |
| BGE，冻结 | 0.4583% | 1.4660% | 2.1709% | 651 |
| BGE + 热度 | 0.4711% | 1.5069% | 2.1826% | 684 |
| BGE + Adapter | 0.5209% | 1.6707% | 2.4661% | 746 |
| BGE + Adapter + 热度 · GPU 精确 / CPU FlatIP | 0.5222% | 1.7074% | 2.5338% | 786 |
| BGE + Adapter + 热度 · CPU IVFFlat | 0.4948% | 1.6010% | 2.4109% | 734 |

| Top 100 点击分层 | 拟合期见过，12,679 对 | 拟合期未见，32,004 对 |
|---|---:|---:|
| V1 热门 | 272 | 0 |
| V1 RRF | 173 | 1 |
| BGE + Adapter | 72 | 674 |
| BGE + Adapter + 热度 · 精确 | 142 | 644 |
| BGE + Adapter + 热度 · IVF | 118 | 616 |

“见过”指笔记在召回拟合期的曝光或请求附带的近期历史中出现过。热度为精确语义检索增加 70 个已见点击命中，同时减少 30 个未见点击命中。V1 双塔曾出现早停样本过小、训练与验证的批内准确率差距大的问题。用相同 float32 向量将 V1 的 IVF 换成全库精确打分，命中仅从 8 增至 10。上述结果支持“稀疏 ID 和近期兴趣表示限制了全库泛化”的解释，训练数值精度故障缺少对应证据。[V1 训练与索引诊断](recall_diagnosis.md) · [V2 消融与可见性](../results/final/evidence.json)

| 候选及排序 | 召回 Top 100 命中 | Top 50 命中 | 最终 Top 10 命中 | 全库 NDCG@10 |
|---|---:|---:|---:|---:|
| V1 RRF → LambdaRank → DIN | 174 | 140 | 78 | 0.001793 |
| V2 精确，直接截断 | 786 | 543 | 178 | 0.003533 |
| V2 精确 → 冻结两级排序，α=0.75 | 786 | 540 | 183 | 0.003473 |
| V2 IVF → 相同两级排序，α=0.75 | 734 | 503 | 172 | 0.003279 |

原排序器在各阶段将模型名次与召回名次融合。精确检索加两级排序后比直接截断多找回 5 个点击，NDCG@10 差值为 −0.000060，按用户配对的 95% 区间为 [−0.000356, +0.000226]。在两端都使用相同排序器时，IVF 相对精确检索的最终 NDCG@10 差值为 −0.000194，区间为 [−0.000384, −0.000019]。这些区间来自 4,576 个用户上的 2,000 次自助抽样。[下游排序回放](../results/final/evidence.json)

## FAISS 的速度与质量

将 `0.03 × log1p(拟合期衰减点击量)` 追加到笔记向量末维，用户向量末维补 1，513 维内积即可同时计算语义与热度。目录中有 3,412 篇同时缺少有效文本与非零热度，与原精确实现按同一规则排除，实际入索引 1,980,526 篇。三种引擎共享用户 Adapter、已交互过滤、同分规则和空历史热门回退。IVFFlat 的 1,024 个倒排桶使用 65,536 篇目录笔记训练 12 轮，每请求探测 64 桶。此配置在测试前固定，仅测这一个 IVF 设置。[检索实现](../src/recall/content_ann.py) · [索引配置](../configs/content_ann.yaml) · [FAISS 索引说明](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

| 引擎 | P50 (ms) | P95 (ms) | 构建或装载张量 (s) | 索引文件 (GiB) | 主机 USS 增量 (GiB) | GPU 张量增量 (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| FAISS FlatIP · CPU | 127.25 | 143.43 | 4.66 | 3.793 | 3.800 | — |
| FAISS IVFFlat · CPU | 10.55 | 11.99 | 34.30 | 3.809 | 3.805 | — |
| PyTorch 全库扫描 · GPU | 29.47 | 29.95 | 0.47 | 共用 V2 向量文件 | 1.900 | 1.980 |

- FAISS 1.16.0 使用 Intel Core i9-13900KF，固定 CPU 0–7、8 个 FAISS 线程。PyTorch 2.7.1+cu128 的分块精确扫描使用 RTX 5090。每个引擎预热 16 个请求，再用相同的 128 个排序验证请求逐条查询，batch=1，GPU 调用同步计时。用已编码用户向量计时，覆盖融合、历史过滤及结果返回。编码、网络 I/O、后续排序另计，线上服务延迟尚无观测。
- 构建从预计算的向量开始，FAISS 构建耗时包括 IVF 训练和添加向量，文件写盘另计。索引文件含 FAISS、笔记 ID 映射和热门回退 ID。主机内存取独立进程加载前后的 USS 增量，GPU 行的主机内存包含访问过的输入映射页。IVFFlat 保存全精度向量，两种 CPU 索引的磁盘和内存占用接近。
- 同一 CPU 上，IVF 的 P95 为 FlatIP 的约 **1/12**。按请求计算，IVF 和 GPU 精确检索的 Top 100 集合平均重合 **91.86%**，CPU FlatIP 与 GPU 精确检索重合 **99.99982%**。CPU 与 GPU 精确检索有 10,993 个请求的完整有序列表完全一致，两者四舍五入前的点击指标一致。IVF 少找回 52 个点击，宏平均 Recall@100 差值为 **−0.1229 个百分点**，用户配对 95% 区间为 **[−0.1890, −0.0560]** 个百分点。[完整计时、指纹与置信区间](../results/final/evidence.json)

## 排序训练监督与评估边界

逐个连接精确 V2 的 Top 100 与该请求真实曝光。曝光记录中的点击和曝光未点提供标签，其余候选为未知。训练窗口总共有 235,340 个唯一曝光对和 58,657 个点击对。

| 窗口 | 请求数 | V2 候选对 | 点击 | 曝光未点 | 未知 | 同时含两类标签的请求 | 可构成的组内正负对 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rank_train | 16,048 | 1,604,800 | 1,206 | 1,718 | 1,601,876 | 203 | 454 |
| rank_val | 6,954 | 695,400 | 457 | 703 | 694,240 | 79 | 178 |
| test | 11,115 | 1,111,500 | 786 | 1,323 | 1,109,391 | 158 | 338 |

排序训练窗口的候选中 **99.8178%** 缺少曝光反馈。2,243 个请求至少有一条可用标签，仅 203 个请求同时有点击和曝光未点，涉及 175 个用户、454 个组内正负对。基于这个覆盖量和前述 NDCG 区间，本轮保留原 DIN 与 LambdaRank 权重。未知候选与任何已曝光物品都不会组成监督负对。[反馈审计](../scripts/content_feedback.py) · [按窗口的审计结果](../results/final/evidence.json)

Qilin 官方测试分片已用于 V1 诊断，也影响了 V2 研究方向。本轮沿用验证期确定的 Adapter、热度和排序参数。在共享盘及官方数据仓库中核对了推荐分片，核验时 `main` 仍指向固定版本 `2d79000cef450b99f6423357ce86e5c132d32dd5`，两者仅有同一套训练、测试文件。因而以上是固定快照上的离线回放。笔记目录没有独立上线时间，分层只能衡量拟合期可见性。V2 Adapter 沿用一个种子，用户配对区间只反映测试用户抽样的波动，跨训练种子的方差和线上 CTR 仍未知。[数据审计](data_audit.md) · [分片版本核验](../results/final/window_audit.json) · [Qilin 数据发布目录](https://huggingface.co/datasets/THUIR/Qilin/tree/main)

## Atlas 复现

先按主项目与 V2 报告准备 `runs/release`、`runs/content-v2` 和固定版本 BGE，在仓库中保留已核验的对照文件 `results/content/qilin_comparison.json`。下列 Bash 命令从这些产物复算。一轮监督 BERT 检查点在 `runs/final/bert-fit`，IVF 在 `runs/final/content-ann/ivf`。仓库只存配置、代码和聚合实验结果。[基线与数据准备](../README.md) · [V2 编码和训练](content_retrieval.md#复现) · [重新生成 BGE 对照](qilin_comparison.md#标签实现与复现)

```bash
set -euo pipefail
cd /sharefiles7/geqinglong/projects/QilinRec
source .venv/bin/activate
export QILIN_DATA=/sharefiles7/geqinglong/projects/QilinRank/data CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=4
python -m pip install -r requirements-content.txt
study=runs/final
bert="$study/bert-base-chinese"
rev=8f23c25b06e129b6c986331a13d8d025a92cf0ea
upstream=https://raw.githubusercontent.com/RED-Search/Qilin/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b
hf=${HF_ENDPOINT:-https://huggingface.co}
mkdir -p "$bert" "$study/protocol"
for f in config.json tokenizer.json tokenizer_config.json vocab.txt model.safetensors
do
  curl -fLsS --retry 2 "$hf/google-bert/bert-base-chinese/resolve/$rev/$f" -o "$bert/$f"
done
curl -fLsS "$upstream/baselines/src/evaluator.py" -o "$study/protocol/evaluator.py"
curl -fLsS "$upstream/datasets/recommendation.test.qrels.csv" -o "$study/protocol/qrels.csv"
export CUDA_VISIBLE_DEVICES=1
python -m scripts.fit_bert_reference --run runs/release --data "$QILIN_DATA" --bert "$bert" --out "$study/bert-fit"
python -m scripts.compare_text_baselines --run runs/release --data "$QILIN_DATA" \
  --bert "$bert" --trained-bert "$study/bert-fit" --qrels "$study/protocol/qrels.csv" \
  --scorer "$study/protocol/evaluator.py" --controls results/content/qilin_comparison.json --out "$study/text_complete.json"

export CUDA_VISIBLE_DEVICES=0
python -m scripts.content_feedback --run runs/release --content runs/content-v2 --out "$study/feedback.json"
ann=(taskset -c 0-7 python -m scripts.content_ann --run runs/release --content runs/content-v2 --out "$study/content-ann")
for engine in flat ivf
do
  "${ann[@]}" --engine "$engine" --mode build
  "${ann[@]}" --engine "$engine" --mode retrieve --stage test
done
taskset -c 16-23 python -m scripts.run_pipeline --config configs/default.yaml --run runs/release \
  --data "$QILIN_DATA" --content-model-dir runs/content_models/bge-small-zh-v1.5 \
  --content-out runs/content-v2 --content-ivf "$study/content-ann" --resume
for engine in flat ivf torch
do
  "${ann[@]}" --engine "$engine" --mode latency
done
python -m scripts.report_freeze --run runs/release --content runs/content-v2 --study "$study" \
  --controls results/content/qilin_comparison.json --dest results/final
python -m unittest discover -s tests -v
```

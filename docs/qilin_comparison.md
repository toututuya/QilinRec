# 与 Qilin 论文的推荐排序指标对照

在作者公开配置指定的前 1,000 个推荐测试请求上，QilinRec V2 的 MRR@10 为 **0.5882**，MAP@10 为 **0.4404**。两项均高于论文中的 BM25，分别低于 BERT 双编码器 0.0185 和 0.0144。官方数字来自论文表 10 的 Recommendation 部分，QilinRec 数字来自 Atlas 上的冻结模型回放。[论文 §5.1 与表 10](https://xuanyuan14.github.io/files/SIGIR25Chen.pdf#page=9) · [实测聚合结果](../results/content/qilin_comparison.json)

## 给定官方曝光候选时的排序

| 模型与来源 | MRR@10 | MRR@100 | MAP@10 | MAP@100 |
|---|---:|---:|---:|---:|
| Qilin · BM25 | 0.5379 | 0.5418 | 0.3933 | 0.4634 |
| Qilin · BERT 双编码器 | 0.6067 | 0.6087 | 0.4548 | 0.5183 |
| Qilin · BERT 交叉编码器 | 0.6346 | 0.6362 | 0.4786 | 0.5394 |
| Qilin · DCN-V2 | 0.6307 | 0.6321 | 0.4651 | 0.5278 |
| Qilin · VLM | 0.6394 | 0.6409 | 0.4890 | 0.5477 |
| QilinRec · 原双塔点积 | 0.5597 | 0.5625 | 0.4181 | 0.4869 |
| QilinRec · 单独 DIN | 0.5958 | 0.5988 | 0.4433 | 0.5107 |
| QilinRec · 仅 V2 内容与热度分数 | 0.5948 | 0.5976 | 0.4398 | 0.5074 |
| **QilinRec · V2 两级排序** | 0.5882 | 0.5906 | 0.4404 | 0.5074 |

MRR 衡量首个点击排得多靠前，MAP 还考虑其他点击的位置。前 1,000 个请求共有 18,437 个曝光候选和 5,382 个不同的请求–点击笔记对。原双塔点积、DIN 和 V2 使用同一批候选与标签。V2 在全库任务的验证集上选出的融合权重搬到曝光池后，这批请求上的 MRR@10 和 MAP@10 均低于单独 DIN。论文与本项目的模型输入、训练样本和选择检查点的方法也有差异。表 10 没有列出该次运行的逐请求预测或请求编号，因此这些差值是论文参照值。[公开配置](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/config/recommendation_dpr_config.yaml#L49-L59) · [作者的候选读取实现](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/dataset_factory.py#L150-L160)

## 固定曝光候选的内容消融

四组实验交叉比较两个因素。用户向量取冻结 BGE 编码的近期点击标题，或经过拟合期训练、验证期选定的第 1 轮 Adapter。所有物品共用同一份标题和正文向量。基础分为余弦相似度，加入热度时额外加上 `0.03 × log1p(拟合期时间衰减点击数)`。热度系数沿用正式 V2，同一请求的四组候选与二值点击逐项相同。

| 方法 | 前 1,000 MRR@10 | MAP@10 | NDCG@10 | 全部 11,115 MRR@10 | MAP@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| 纯 BGE | 0.5942 | 0.4296 | 0.5703 | 0.5735 | 0.4527 | 0.5827 |
| BGE + Adapter | 0.5911 | 0.4397 | 0.5770 | 0.5764 | 0.4568 | 0.5860 |
| BGE + 热度 | 0.5974 | 0.4333 | 0.5732 | 0.5769 | 0.4550 | 0.5847 |
| BGE + Adapter + 热度 | 0.5948 | 0.4398 | 0.5778 | 0.5784 | 0.4574 | 0.5869 |
| V2 两级排序 | 0.5882 | 0.4404 | 0.5781 | 0.5819 | 0.4623 | 0.5918 |
| 单独 DIN | 0.5958 | 0.4433 | 0.5806 | 0.5815 | 0.4626 | 0.5914 |

前 1,000 个请求里，纯 BGE 加 Adapter 后，MAP@10 增加 0.0101，首击 MRR@10 下降 0.0032。NDCG@10 增加 0.0067，按用户配对重采样 2,000 次的 95% 区间为 [−0.0018, +0.0145]。完整测试集也包含这前 1,000 个请求，Adapter 的 NDCG@10 差值为 +0.0033，区间为 [+0.0008, +0.0057]。在已有 Adapter 的基础上增加热度，完整测试集的额外差值为 +0.0009，区间为 [−0.0003, +0.0021]。表 10 未报告 NDCG，项目以二值点击、按请求归一化的折损累计增益补充这一指标。[各组分数与配对区间](../results/content/qilin_comparison.json)

## 全部测试请求与全库召回

| QilinRec V2 的任务 | 请求数 | MRR@10 | MAP@10 |
|---|---:|---:|---:|
| 给定曝光候选，配置的前 1,000 个请求 | 1,000 | 0.588238 | 0.440390 |
| 给定曝光候选，完整测试集 | 11,115 | 0.581881 | 0.462271 |
| 从全库召回再排序，完整测试集 | 11,115 | 0.004876 | 0.001774 |

前 1,000 个请求的曝光池大小中位数为 12，完整测试集为 9，最大为 79。全部 44,683 个不同的请求–点击笔记对均已在各自的曝光池中。全库任务则从 1,983,938 篇笔记里取 Top 100，V2 找回 786 个点击，请求宏平均 Recall@100 为 **2.5338%**，按点击对计为 **1.7591%**。最终 Top 10 保留 183 个。表 10 未给出同目录、同预算的召回率。[全库分层与各阶段命中](content_retrieval.md)

| 完整测试集的召回方法 | Recall@10 | Recall@50 | Recall@100 | Top 100 命中数 |
|---|---:|---:|---:|---:|
| 时间衰减热门 | 0.2072% | 0.3641% | 0.4800% | 272 |
| 纯 BGE | 0.4583% | 1.4660% | 2.1709% | 651 |
| BGE + Adapter | 0.5209% | 1.6707% | 2.4661% | 746 |
| BGE + 热度 | 0.4711% | 1.5069% | 2.1826% | 684 |
| BGE + Adapter + 热度 | 0.5222% | 1.7074% | 2.5338% | 786 |

前三列均为按请求宏平均的全库召回率，所有未命中点击保留在分母中。每列在同一套 Top 100 检索结果上按原分数截取，计算位置在 LambdaRank 和 DIN 之前。纯 BGE 命中 651 个，加 Adapter 后为 746 个，再加热度后为 786 个。最后一组的有序 Top 100 与正式归档结果在全部 11,115 个请求上逐篇一致。

## 标签、实现与复现

- 按测试 Parquet 的原始行号连接官方 qrels，再映射到项目内部的时间排序。逐请求核对 `rec_results` 和曝光集合、官方点击和项目正例，保留全部请求和池内物品。曝光池的 MRR、MAP 使用作者评分函数及原始 qrels。二值 NDCG 和全库主实验对每个请求–物品点击计一次。完整标签有两条重复记录，首 1,000 个请求没有重复标签。统一去重后，两种实现的 MRR、MAP、Recall 一致，NDCG 另用逐请求实现复核。[作者评分函数](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/evaluator.py#L16-L90) · [官方标签](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/datasets/recommendation.test.qrels.csv)
- 前 1,000 个请求有 6 个缺少可编码的近期标题，完整测试集有 65 个。此时纯语义组在曝光池内按笔记 ID 打破平局，在全库检索中返回空列表。加热度的两组均回退到历史热门。有用户文本时，缺少物品文本的候选在纯语义排序中排在有效候选之后，加热度时仍可用拟合期热度打分。模型打分忽略原曝光名次。
- 复用原检查点、输入和验证期冻结的权重。曝光池里的每个候选都可以参与排序，包括历史中已见的候选。用内容与热度分数排列初始候选，LambdaRank 参与选取前 50，DIN 参与短名单排序。为计算 @100，将余下候选按粗排名次接在短名单后。全部 11,115 个请求的前 10 个输出均与项目原有排序函数在同一输入上的结果一致。
- 曝光池主结果在 RTX 4090 上计算，全库消融在 RTX 5090 上计算。用同一组权重在 RTX 5090 复算前 1,000 个请求的曝光排序，V2 的 MRR@10 同为 0.588238，MAP@10 从 0.440390 变为 0.440365，远小于与论文 BERT 双编码器的 0.0144 差距。这项对照只测了推理设备带来的差异。完整测试上，在 RTX 4090 复算的原 DIN 去重指标与既有正式记录相符。

从已有 `runs/release` 与 `runs/content-v2` 复算。脚本校验作者评分代码和 qrels 的 SHA-256，输出仅包含聚合指标。单卡运行时设置对应的 `CUDA_VISIBLE_DEVICES` 并传 `--device cuda`，两项评估会依次使用该卡。[正式产物准备](content_retrieval.md#复现) · [回放脚本](../scripts/compare_qilin.py)

```bash
cd /sharefiles7/geqinglong/projects/QilinRec
source .venv/bin/activate
export QILIN_DATA=/sharefiles7/geqinglong/projects/QilinRank/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
rev=299c02667e1dc98dd8e62aa5e13a8e04a9d0363b
root="https://raw.githubusercontent.com/RED-Search/Qilin/$rev"
out=runs/official_comparison
mkdir -p "$out"
curl -fLsS "$root/baselines/src/evaluator.py" -o "$out/evaluator.py"
curl -fLsS "$root/datasets/recommendation.test.qrels.csv" -o "$out/qrels.csv"
python -m scripts.compare_qilin --run runs/release --content runs/content-v2 \
  --data "$QILIN_DATA" --qrels "$out/qrels.csv" --scorer "$out/evaluator.py" \
  --out "$out/comparison.json" --device cuda:0 --catalog-device cuda:1
```

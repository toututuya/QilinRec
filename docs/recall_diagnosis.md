# QilinRec 召回与模型选择复核

本次在 Atlas 复用正式运行 `runs/release`，对已发布的预测逐请求回放，再用三个种子重新训练双塔。正式测试含 11,115 个请求、44,683 个去重的请求–点击笔记对。双塔的模型选择和融合配额的选择均使用测试之前的窗口。下文的命中数都指请求–笔记对，同一笔记可在多个请求中被点击。

## 官方评测使用的候选池

Qilin 论文第 3 节说明，数据在请求发起前采集最近 20 条点击笔记。第 5.1 节把这些笔记的标题拼成用户兴趣文本，再用它匹配候选笔记的标题和正文。公开 DCN 数据处理器直接取近期列表的前 10 项。推荐 DPR 配置载入预训练中文 BERT，指定 `evaluate_type: rerank` 和 `results_key: rec_results`。该配置还设定 `sample_num: 1000`，对应数据处理器取测试文件的前 1,000 行。这是发布代码的默认值，论文表 10 的请求数需以作者该次运行记录为准。[论文 §§3、5.1 及表 10](https://xuanyuan14.github.io/files/SIGIR25Chen.pdf) · [推荐 DPR 配置](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/config/recommendation_dpr_config.yaml) · [重排候选读取与截取](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/dataset_factory.py#L150-L200) · [DCN 历史输入](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/dataset_factory.py#L480-L495)

按测试 Parquet 原始行号连接官方 qrels。11,115 个请求的 `rec_results` 均与各自去重曝光集合相同，总计 179,401 个候选。每个请求的最少、中位和最多候选数分别为 1、9、79。官方 qrels 的 44,685 行去重后与全部 44,683 个已观测点击逐请求一致。单是拿到官方候选列表，Recall@100 就已经等于 1。全目录任务需要先从 1,983,938 篇笔记中找回这些点击。[官方 qrels](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/datasets/recommendation.test.qrels.csv) · [逐请求核对汇总](../results/diagnosis/frozen_audit.json)

| 曝光候选排序 | MRR@10 | MAP@10 | 数据来源 |
|---|---:|---:|---|
| BERT 双编码器 | 0.6067 | 0.4548 | Qilin 论文表 10 |
| BERT 交叉编码器 | 0.6346 | 0.4786 | Qilin 论文表 10 |
| DCN-V2 | 0.6307 | 0.4651 | Qilin 论文表 10 |
| QilinRec 双塔点积 | 0.5509 | 0.4408 | 本次回放全部测试请求 |
| QilinRec DIN | 0.5815 | 0.4626 | 正式模型，全部测试请求 |

论文数值仅作参照。QilinRec 另设了严格的时间窗口，训练请求、特征与模型规模均有差异。小红书公开课介绍了工业多路召回和后续排序，GemiRec 作者报告了首页多兴趣召回的两周线上 A/B 实验。本项目的回放只能评价日志中有标签的点击，线上收益需要新的曝光反馈来验证。[王树森公开课讲义](https://raw.githubusercontent.com/wangshusen/RecommenderSystem/main/Slides/01_Basics_02.pdf) · [GemiRec §5.7](https://arxiv.org/html/2510.14626v1#S5.SS7)

## 固定预算时，点击在哪一步丢失

| 正式测试召回 | 每请求候选数 | 命中点击 | 请求宏平均 Recall | 点击微平均留存 |
|---|---:|---:|---:|---:|
| 热门 | 100 | 272 | 0.4800% | 0.6087% |
| ItemCF | 最多 100 | 21 | 0.0390% | 0.0470% |
| 双塔 IVFFlat | 100 | 8 | 0.0251% | 0.0179% |
| 三路等权 RRF | 100 | 174 | 0.3504% | 0.3894% |
| 三路完整并集 | 199–300，平均 269.85 | 301 | 0.5440% | 0.6736% |
| 热门，逐请求与并集等长 | 与上一行逐请求相同 | 469 | 0.8511% | 1.0496% |
| 热门 | 300 | 505 | 0.9291% | 1.1302% |

三路并集保住热门的 272 个点击，另外带来 29 个。RRF 截到 100 时挤掉了 105 个热门命中，只补回 7 个，留下 174 个。合并后再用 RRF 排到 300 会保留同一批物品，对这三路而言完整并集与 RRF@300 的召回相等。让粗排处理更大的并集可以保留更多已知点击，但需要同时报告等长热门和最终 Top 50/10，才能量化额外计算带来的收益。[融合分解](../results/diagnosis/frozen_audit.json) · [等预算回放](../results/diagnosis/review_replay.json)

另在较早的 `rank_train` 试验为热门保留 0、25、50、75、90、100 个名额，其余名额由原 RRF 补齐，维持每请求 100 个。按请求宏平均 Recall 选择了 100。在随后的 `rank_val`，原 RRF、保留 90 个、全热门分别找回 94、143、159 个点击。这项选择只优化日志点击找回率。正式测试上，纯热门覆盖全目录 0.0059%，三路 RRF 覆盖 6.1698%。覆盖率和点击命中数的取舍需单独报告。[配额验证](../results/diagnosis/frozen_audit.json)

## 双塔有学习信号，全库泛化较弱

正式模型的前五轮训练损失为 5.968、5.826、5.691、5.590、5.481。原来仅用 768 个固定请求选轮次，各轮 Top 100 命中数为 **0、1、2、1、1**。获选的第 3 轮在全部 5,563 个召回验证请求命中 10 个，最后的第 5 轮命中 16 个。在更晚的排序验证窗口，两个冻结检查点用精确全库打分分别命中 9 和 12 个。原来的早停取样确实丢失了模型选择信号。[原始训练轨迹](../results/formal/evidence.json) · [检查点回放](../results/diagnosis/frozen_audit.json)

在全部正式测试请求上复用相同的 float32 查询和物品向量、相同的已交互过滤，以 FAISS FlatIP 穷举全目录，Top 100 找回 10 个点击。IVFFlat 找回 8 个，均包含在前者的命中中。两种索引的 Top 100 集合平均重合 90.99%。物品向量范数在 0.9999998 至 1.0000002 之间，查询和物品均为有限值。精确打分只补回 2 个点击，主要损失已经发生在模型给出的排序中。[全测试索引回放](../results/diagnosis/frozen_audit.json)

正式模型把当前请求附带的近期点击用于过滤，送入用户塔的是从已结束的训练请求重建的历史。测试时，重建历史在 9,038 个请求非空，官方历史在 11,050 个请求非空。按请求计算，两者集合的平均 Jaccard 相似度只有 7.52%，拟合窗口则为 53.85%。这些近期列表只有集合采样时点的保证，逐条点击的精确时间未知。[历史审计](../results/diagnosis/frozen_audit.json)

配对实验固定 35,424 个拟合请求、119,465 个点击对、词表、网络、采样、过滤和优化器。每个种子分别训练重建历史与官方近期列表两组，各 12 轮，使用相同初始化和批次顺序。每轮对全部 5,563 个 `tower_val` 请求精确检索，按请求宏平均 Recall@100 选轮次。下表在随后 6,954 个 `rank_val` 请求、22,579 个点击上复核获选模型。双塔重训和选择均未使用正式测试标签。

| 随机种子 | 重建 / 官方获选轮次 | 重建 / 官方命中 | 重建 / 官方宏平均 Recall@100 |
|---|---:|---:|---:|
| 20261127 | 10 / 11 | 15 / 29 | 0.1159% / 0.1641% |
| 20261128 | 5 / 5 | 14 / 16 | 0.1021% / 0.0881% |
| 20261129 | 10 / 4 | 16 / 17 | 0.1029% / 0.0563% |
| 三种子算术均值 | — | 15.0 / 20.7 | 0.1070% / 0.1028% |

替换历史提高了三个种子的命中总数，宏平均只在一个种子上提高。一个点击在只点一篇的请求中和在点击多篇的请求中，对宏平均有不同贡献。同一窗口的热门基线找回 159 个点击。固定在第 12 轮读取时，重建 / 官方历史分别平均命中 13.7 / 24.3 个，宏平均为 0.0923% / 0.1701%。这一行用于观察训练时长的影响，以完整验证窗口选择的结果见上表。[六次训练的逐轮记录](../results/diagnosis/paired_training.json)

用相同的批内对比题检查训练与验证。每次固定抽取 4,608 对，每批 384 个候选，并按原训练规则屏蔽重复和已知正例。到第 12 轮，各设置在训练样本的批内 Top 1 为 21.0%–22.4%，在验证样本为 2.0%–3.3%，均匀随机约为 0.26%。模型学到了训练样本的区分信号，但延长训练留下很大的泛化差距。批内准确率也不能代替近 200 万物品的 Recall。[批内对照](../results/diagnosis/paired_training.json)

## 冷启动和曝光标签覆盖

正式测试有 32,004 个点击对的物品在召回拟合期的曝光与近期历史中均未出现，占 71.62%。在同一全目录输出上，仅分层计算命中，避免更换目标集合时一起改变检索难度。

| 目标物品在拟合期是否出现 | 点击对分母 | 热门@100 | ItemCF@100 | 双塔@100 | RRF@100 | 三路完整并集 |
|---|---:|---:|---:|---:|---:|---:|
| 出现过 | 12,679 | 272 | 21 | 1 | 173 | 294 |
| 未出现 | 32,004 | 0 | 0 | 7 | 1 | 7 |

这份日志没有独立的笔记上线时间。这里的冷启动按拟合期可见性定义。另有 37,507 个测试点击对的物品在拟合期没有点击记录，36,429 个使用双塔的共享未知 ID 表示。模型以 24 维向量编码物品，从头学习两级类目和最多 6 个标题字符片段。7 次未见物品命中说明双塔提供了少量热门无法给出的结果，也说明标题、正文及用户近期兴趣的语义表示值得独立验证。[分层召回](../results/diagnosis/review_replay.json) · [正式实验](../results/formal/report.md)

按历史请求重新取 RRF Top 100 并连接曝光标签。在 16,048 个排序训练请求生成的 1,604,800 个请求–候选对中，仅 317 个已点击、585 个已曝光未点，其余 1,603,898 个没有标签。只有 24 个请求同时具有点击和已曝光未点。直接把训练集裁成此交集会使 LambdaRank 可用的成对监督极少。原工程已经同时报告完整曝光排序与全链路排序，生成候选若用于训练，需同时报告真实反馈覆盖情况，保留未曝光物品的未知状态。[候选监督覆盖](../results/diagnosis/review_replay.json)

## LambdaRank 和 DIN

训练回调先按请求、分数降序、同分物品 ID 升序排列标签。折损权重按每个请求内的第 1、2、… 个输出位置生成，组边界在排序后保持，位置权重无需再应用样本排列。独立参考实现逐请求用显式排序和 DCG/IDCG 复算所有 45 个树前缀，覆盖 6,954 个验证请求、95,731 个曝光对。与训练记录的最大绝对差为 2.22 × 10⁻¹⁶，最优均为 **第 25 棵树，NDCG@10 = 0.598938252945**。这支持保留原模型选择。[独立逐轮复算](../results/diagnosis/review_replay.json) · [训练指标实现](../src/prerank.py)

原 RRF Top 100 留有 174 个测试点击。LambdaRank 在 Top 50 留下 140 个，RRF 直接截断留下 128 个。MLP 和 DIN 在相同的 50 个候选中分别把 59、78 个点击排入 Top 10。完整曝光列表上，DIN 的 NDCG@10 比 MLP 高 0.000676，用户配对自助抽样的 95% 区间为 [−0.001384, +0.002788]。DIN 的曝光 LogLoss 更低，AUC 略低。二者验证 AUC 在第 3 轮和第 2 轮分别达到峰值，继续训练的验证 LogLoss 上升。精排有局部收益，在已丢失正例的请求上没有重新引入物品的入口。[正式结果和置信区间](../results/formal/report.md)

## 后续实验的优先级

优先检验冷启动笔记的内容匹配。用官方近期点击标题和候选的标题、正文训练或评估中文语义双编码器，固定目录、切分、过滤与候选预算。在更晚的独立时间窗口，同时看请求宏平均、命中数及拟合期可见性分层。完整并集进入粗排适合作为容量对照，与逐请求等长的热门候选共用排序器，比较最终 Top 50/10 及计算开销。官方测试集已用于诊断此次基线，后续若要独立确认收益，需预先锁定方案并保留新的时间窗口。

## 在 Atlas 复现

以下命令从已有正式运行重放。新的配对模型和数据索引链接写入 `runs/diagnosis-repro`，该目录与正式运行位于同一文件系统。汇总结果存入仓库，Atlas 的本次配对模型位于 `runs/retrieval_diagnosis/paired/{reconstructed,official_recent}/<seed>/base/`。完整数据准备、原始模型训练和依赖版本见项目 README。

```bash
source .venv/bin/activate
export QILIN_DATA=/sharefiles7/geqinglong/projects/QilinRank/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p runs/diagnosis-repro
curl -fL https://raw.githubusercontent.com/RED-Search/Qilin/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/datasets/recommendation.test.qrels.csv -o runs/diagnosis-repro/official-qrels.csv
echo '3ffc5bc91420992686836ec5b51cfe725e79652140ef4168b0bcf81433797525  runs/diagnosis-repro/official-qrels.csv' | sha256sum -c -
python -m scripts.diagnose_recall --run runs/release --data "$QILIN_DATA" --qrels runs/diagnosis-repro/official-qrels.csv --out runs/diagnosis-repro/frozen_audit.json
python -m scripts.diagnose_tower --run runs/release --work runs/diagnosis-repro/paired --out runs/diagnosis-repro/paired_training.json
python -m scripts.audit_review --run runs/release --out runs/diagnosis-repro/review_replay.json
```

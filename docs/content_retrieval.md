# 用中文内容向量改善全库召回

在 Atlas 上用 1,983,938 篇笔记构建内容索引。在相同的 11,115 个官方测试请求上，每请求返回 100 篇，新通道找回 786 个已观测点击，时间衰减热门找回 272 个，原三路等权 RRF 找回 174 个。总分母为 44,683 个去重的请求–点击笔记对。下游沿用原有 LambdaRank 和 DIN 检查点，在排序中保留一部分召回顺序后，最终 Top 10 找回 183 个点击。[实验表](../results/content/experiments.csv) · [配置、逐轮指标、检查点指纹及区间](../results/content/evidence.json)

## 改了哪些输入和评分

- **物品塔**读取笔记标题和正文，使用固定版本的中文 BGE-small-zh-v1.5，取归一化的 CLS 向量。每篇最多 64 个模型 token，正文在分词前最多读 512 个字符。512 维向量以 FP16 保存，检索时还原为 FP32、归一化并穷举全目录。模型权重和约 2.03 GB 的物品向量保存在 Atlas。[BAAI 模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/7999e1d3359715c523056ef9478215996d62a620) · [作者的编码与池化实现](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/encoder_only/base.py)
- **用户塔**使用官方请求前近期点击列表的前 10 个笔记 ID，按给定顺序拼接非空标题，最多编码 128 个 token。11,050 个测试请求有可用文本。空历史在融合策略中回退到热门。给定官方近期点击历史的时点及其用于构造兴趣文本的依据见 [Qilin 论文 §§3、5.1](https://xuanyuan14.github.io/files/SIGIR25Chen.pdf)。
- **监督训练**冻结 BGE，在用户向量上加一个 96 维瓶颈的残差网络，输出重新归一化。用 119,031 个有可用文本的拟合期点击对训练 4 轮，批大小 384。批内对比沿用重复物品、请求其他正例和已知历史的屏蔽规则。第 0 轮是冻结 BGE 对照。在全部 5,563 个召回验证请求中，第 0–4 轮分别命中 247、268、277、258、257 个。按请求宏平均 Recall@100 选择第 1 轮，宏平均为 2.0560%。
- **融合**用内容余弦相似度加上 `β × log(1 + 拟合窗口末的衰减点击量)`。候选和热度均排除用户已经交互的笔记。实现先取全目录语义 Top 100，再对所有热度大于零的目录物品打融合分，合并去重后取 Top 100。因热度修正非负，其他物品的分数不会超过原语义 Top 100 的门槛。相同得分按笔记 ID 升序，单测与逐项穷举对照。
- **下游排序**使用原 LambdaRank、MLP、DIN 的权重和原来的 19 维特征。另为 LambdaRank 选 Top 50 和 DIN 选 Top 10 增加相同的排序融合权重，在各自候选中按 `α/(60+召回名次) + (1−α)/(60+模型名次)` 排序。α=0 复现冻结排序器，α=1 直接保留召回顺序。预测物品逐请求检查去重、已交互排除、目录成员和 100 → 50 → 10 的包含关系。

## 验证如何决定参数

按时间顺序使用原有窗口。在 16,048 个 `rank_train` 请求上比较纯热门及 β ∈ {0, 0.03, 0.08, 0.16, 0.32, 0.64}，以请求宏平均 Recall@100 选 β=0.03。β=0.08 的总命中更多，但宏平均较低。随后在 6,954 个 `rank_val` 请求复核，原 RRF、热门、纯语义和所选融合分别找回 **94、159、441、457** 个点击。所选融合宏平均为 2.9218%，同时超过两项基线后锁定。纯热门或新方案都只给下一层 100 个名额。

同一验证窗口上，融合召回直接截 Top 10 找回 114 个，照搬原粗精排后为 56 个。观察到冻结排序器丢失内容命中后，追加排序顺序融合对照。以请求开始时间中位点 1732703134 切分该窗口，2,882 个此前已结束的请求用于在 {0, 0.25, 0.5, 0.75, 1} 中选择 α，保留 3,478 个随后开始的请求用于复核，剔除 594 个跨界请求。早段按 NDCG@10 选出 α=0.75。后段冻结排序、α=0.75、直接召回分别找回 38、68、61 个点击，NDCG@10 为 0.003041、0.005843、0.005888。所选值通过相对冻结排序的确认条件。全部官方测试结果用于回放描述，本次沿用了诊断阶段已经检视过的测试分片。[选择和验证记录](../results/content/evidence.json)

## 正式测试

| 候选来源或排序方式 | Top 100 命中 | Top 50 命中 | Top 10 命中 | Top 10 NDCG |
|---|---:|---:|---:|---:|
| 原 RRF → LambdaRank → DIN | 174 | 140 | 78 | 0.001793 |
| 热门 → 同一 LambdaRank → 同一 DIN | 272 | 172 | 78 | 0.001809 |
| 新融合直接按召回分数截断 | 786 | 543 | 178 | 0.003533 |
| 新融合 → 原 LambdaRank → 原 DIN | 786 | 393 | 75 | 0.001553 |
| 新融合 → 两级排序均保留召回顺序，α=0.75 | **786** | **540** | **183** | **0.003473** |

- 新融合的请求宏平均 Recall@100 为 **2.5338%**，纯热门为 0.4800%，绝对提升 2.0539 个百分点。按用户做 2,000 次配对自助抽样，95% 区间为 [+1.7248, +2.4423] 个百分点。新融合的微平均留存为 1.7591%，目录覆盖率为 26.06%。
- 在拟合期曝光和近期历史均未出现的 32,004 个测试点击对中，纯热门命中 0 个，纯语义命中 674 个，新融合命中 **644 个**。在其余 12,679 个点击对中，三者分别命中 272、72、142 个。加入热度改善总体命中，也付出了 30 个未见物品命中的代价。
- 最终 Top 10 的主要增益来自召回。相对原全链路，NDCG@10 差值为 +0.001681，用户配对 95% 区间为 [+0.000833, +0.002557]。与新召回直接截断相比，两级融合多命中 5 个点击，NDCG@10 差值为 −0.000060，95% 区间为 [−0.000356, +0.000226]。本次结果支持内容召回的收益，对排序相对直接截断的额外增益仍有较大不确定性。

所有召回和全链路指标都保留完整点击分母。BGE 只读取目录快照中的文本，热度仅来自拟合窗口。在这份日志中无法逐请求核实笔记的上线时间，也缺少对未曝光候选的反事实反馈。一次固定种子的离线回放无法推断线上 CTR。

## 复现

Atlas 使用 Python 3.12、PyTorch 2.7.1+cu128 和物理 GPU 0（RTX 5090）。现有项目虚拟环境已有所需依赖。在新环境中先按项目 README 安装兼容的 CUDA PyTorch，再运行 `python -m pip install -r requirements-content.txt`。下载固定版本 BGE 时可通过 `HF_ENDPOINT` 指定 Hugging Face 镜像。编码器启动时验证权重 SHA-256 和五个配置及分词文件的 Git blob 指纹。

```bash
cd /sharefiles7/geqinglong/projects/QilinRec
source .venv/bin/activate
export QILIN_DATA=/sharefiles7/geqinglong/projects/QilinRank/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=4
MODEL=runs/content_models/bge-small-zh-v1.5
REV=7999e1d3359715c523056ef9478215996d62a620
mkdir -p "$MODEL"
for name in config.json tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json model.safetensors
do
  curl -fLsS --retry 2 "${HF_ENDPOINT:-https://huggingface.co}/BAAI/bge-small-zh-v1.5/resolve/$REV/$name" -o "$MODEL/$name"
done
python -m unittest discover -s tests -v
python -m scripts.run_pipeline --config configs/smoke.yaml --run runs/smoke_release --resume --content-model-dir "$MODEL" --content-out runs/repro-content-smoke
python -m scripts.run_pipeline --config configs/default.yaml --run runs/release --resume --content-model-dir "$MODEL" --content-out runs/repro-content
python -m scripts.report_content --run runs/release --out runs/repro-content --dest results/repro-content
```

正式产物在 Atlas 的 `runs/content-v2/`。目录编码按 8,192 行写入恢复进度，用户塔按轮保存，`proposed.json`、`policy.json` 和 `alignment_policy.json` 记录三个选择时点。每层候选、分数、权重和原始文本都留在服务器，仓库交付可复现代码和聚合结果。

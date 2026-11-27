# 源码调研

QilinRec 从推荐请求的曝光明细构建监督数据。下面列出阅读的上游核心实现和由此作出的工程选择。项目代码独立实现。

| 模块 | 上游核心源码 | 上游如何工作 | QilinRec 的选择 |
|---|---|---|---|
| Qilin 数据 | [官方分析脚本](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/datasets/analyze_qilin.py)、[SIGIR 论文](https://arxiv.org/abs/2503.00501) | 脚本从明细时间戳提取请求起点，论文区分近期点击、曝光和后续反馈 | 审计每个请求的最早和最晚时间。标签合并到唯一 request–note 粒度 |
| Qilin 基线 | [dataset_factory.py](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/dataset_factory.py)、[模型](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/model_factory.py)、[训练](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/trainer.py)、[评测](https://github.com/RED-Search/Qilin/blob/299c02667e1dc98dd8e62aa5e13a8e04a9d0363b/baselines/src/evaluator.py) | collate 组装正负例，部分训练器从目录补负例。密集检索用交叉熵，评测区分检索和给定列表重排 | 采样对比只用于召回。排序监督只读已曝光物品的 click。项目单独实现指标并保留每个请求的全部点击真值 |
| Two-Tower | [TensorFlow Recommenders 检索损失](https://github.com/tensorflow/recommenders/blob/main/tensorflow_recommenders/tasks/retrieval.py)、[批量索引与排除项](https://github.com/tensorflow/recommenders/blob/main/tensorflow_recommenders/layers/factorized_top_k.py) | 查询与候选矩阵相乘，沿候选轴做温度缩放与交叉熵，可屏蔽重复正例 | PyTorch 双塔共用物品表示，屏蔽相同物品、当前请求其他正例与已知历史，单测覆盖重复 ID |
| ItemCF | [implicit 的 item–item fit/recommend](https://github.com/benfred/implicit/blob/main/implicit/nearest_neighbours.py) | 稀疏输入、近邻索引、聚合及过滤已交互 ID | 用户活跃度加权，分块稀疏乘法与逐行截断，离线存邻接表 |
| FAISS | [IndexFlat.cpp](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexFlat.cpp)、[IndexIVFFlat.cpp](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexIVFFlat.cpp) | Flat 对全量向量求内积。IVF 由粗量化器分配倒排桶，仅在探测到的桶中比较原向量 | 同一组归一化向量和查询比较 FlatIP 与 IVFFlat 的召回、近邻重合率和耗时 |
| LambdaRank | [LightGBM 的组内梯度](https://github.com/lightgbm-org/LightGBM/blob/main/src/objective/rank_objective.hpp) | 在同一 query 内用标签顺序不同的样本对及交换引起的 NDCG 变化产生梯度 | 连续排列同一 request 的已曝光物品，传入真实 group sizes，在线候选由同一个模型打分 |
| DIN | [作者的模型](https://github.com/zhougr1993/DeepInterestNetwork/blob/master/din/model.py)、[训练循环](https://github.com/zhougr1993/DeepInterestNetwork/blob/master/din/train.py)、[论文](https://arxiv.org/abs/1706.06978)、[PyTorch 模型](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/din.py)、[池化](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/layers/sequence.py)、[激活单元](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/layers/core.py) | 作者代码分别以目标物品激活历史。PyTorch 参考实现支持序列掩码及可选归一化 | 目标/历史、差与积进入激活单元。同一批次接口支持均值池化 MLP 与注意力 DIN，输出 logit，训练时使用 BCEWithLogitsLoss |

## 来源与许可

Qilin 官方 GitHub 树固定为 `299c02667e1dc98dd8e62aa5e13a8e04a9d0363b`。该树根目录未提供 LICENSE。[数据卡](https://huggingface.co/datasets/THUIR/Qilin)在 License 节标注 MIT。作者的 DIN 演示仓库根目录也未列独立许可证。本项目仅引用并说明这些实现，不分发原始数据和上游源码。TensorFlow Recommenders 与 DeepCTR-Torch 采用 Apache-2.0，implicit、FAISS 和 LightGBM 采用 MIT。以各仓库的许可证文本为准。[TFRS](https://github.com/tensorflow/recommenders/blob/main/LICENSE) · [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch/blob/master/LICENSE) · [implicit](https://github.com/benfred/implicit/blob/main/LICENSE) · [FAISS](https://github.com/facebookresearch/faiss/blob/main/LICENSE) · [LightGBM](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE)

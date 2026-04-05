# DiFiNet Baseline 改动说明（相对官方源码）

本说明用于总结当前仓库相对官方 DiFiNet 的结构改动、保留策略与复现方式。

## 1. 改动目标

在不破坏官方主干的前提下，对关键模块做可插拔增强，形成可对比的 baseline 框架。

## 2. 相对官方的主要改动模块

### 2.1 SDM（Self-adaptive Differentiation Module）

- 官方：`hard argmax + gumbel` 单一路径。
- 当前：支持多种掩码策略，可切换：
  - `hard_gumbel`
  - `soft_topk`
  - `soft_topk_mix`
  - `softmax`

代码位置：
- `model/cnn_liabrary.py`
- `model/cnn.py`

### 2.2 BFM（Boundary Filtration Module）

- 官方：原始 `legacy` 过滤分支。
- 当前：支持多种 BFM 结构：
  - `legacy`（官方）
  - `pyramid`
  - `pyramid_gated`
  - `pyramid_gated_se`

代码位置：
- `model/cnn.py`

### 2.3 最终打分头（Decoder Score Head）

- 官方：单层线性头。
- 当前：新增 `residual_mlp` 头（线性主分支 + 残差 MLP 分支），用于增强表达能力且保持训练稳定。

代码位置：
- `model/model.py`

### 2.4 Refiner 可选增强（保留）

- 官方：`maskcnn`。
- 当前：新增 `dilated` 轻量残差空洞卷积分支作为可选增强。

代码位置：
- `model/cnn.py`
- `model/model.py`

### 2.5 Span Semantic Encoder 可选增强（保留）

- 官方：词级聚合后直接进入后续打分。
- 当前：新增 `span_encoder_type=dwconv_res`，在词级表示后加入轻量局部上下文残差增强（depthwise + pointwise conv）。

代码位置：
- `model/model.py`
- `train.py`

## 3. 已清理（不保留）的无效增强

以下模块在 weibo 对比中未带来稳定收益，已从当前代码移除：

- `fusion_type=channel_gate`
- `fusion_type=residual_gate`
- `loss_type=asymmetric_focal`（及其 `asl_gamma_*` 参数）

## 4. 当前推荐主框架（保留）

主框架（历史 best）：

- `bfm_type=pyramid_gated`
- `sdm_mask_type=soft_topk_mix`
- `head_type=residual_mlp`
- 常用搭配：`pair_scorer=biaffine`，`fusion_type=sum`，`refiner_type=maskcnn`

在本仓库历史 5 epoch weibo 实验中，对应 `max test F1 = 73.55`。

## 5. 指标口径说明（重要）

同一配置下会出现两个常见口径：

- `metric.log`：按 epoch 全程看，取 `test` 最大值（`max test`）
- `best_metric.log`：按 `dev` 最优 checkpoint 选点，再读取对应 `test`

示例（同一配置）：

- `logs/log_20260405_185636/metric.log` 的 `max test = 73.55`（epoch 5）
- `logs/log_20260405_185636/best_metric.log` 对应 `test = 72.24`（dev 最优发生在 epoch 4）

因此 `73.55` 与 `72.24` 来自同一配置，只是选模口径不同。

## 6. 复现实验命令（weibo）

### 6.1 主框架（历史 best 配置）

```bash
python train.py \
  -n 5 -d weibo \
  --model_name /mnt/workspace/pretrained_models/google-bert/bert-base-chinese \
  --lr 2e-5 --cnn_dim 120 -b 2 --accumulation_steps 8 \
  --cnn_depth 1 --n_head 4 --logit_drop 0.1 --seed 42 --n_layer 1 \
  --size_feature_type bias --word_pooling mix --pool_gate_type scalar \
  --refiner_type maskcnn --span_encoder_type none \
  --pair_scorer biaffine --mlp_type mlp --fusion_type sum \
  --head_type residual_mlp \
  --bfm_type pyramid_gated --sdm_mask_type soft_topk_mix --sdm_topk 2
```

### 6.2 可选增强 A：Refiner 切换到 dilated

```bash
python train.py ... --refiner_type dilated
```

### 6.3 可选增强 B：Span Encoder 增强

```bash
python train.py ... --span_encoder_type dwconv_res
```

## 7. 关键代码入口

- 训练参数入口：`train.py`
- 主模型与打分头：`model/model.py`
- SDM/BFM 与 Refiner：`model/cnn.py`
- SDM 掩码策略：`model/cnn_liabrary.py`


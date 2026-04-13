# DiFiNet Baseline 改动说明（相对官方源码）

本说明用于总结当前仓库相对官方 DiFiNet 的结构改动、保留策略与复现方式。

## 1. 改动目标

在不破坏官方主干的前提下，对关键模块做可插拔增强，形成可对比的 baseline 框架。

## 1.1 代码收敛（2026-04-13）

按“保留最优框架、移除无效分支”的策略，训练与模型代码已固定为单一路径：

- `bfm_type=pyramid_gated`
- `sdm_mask_type=soft_topk_mix`（`sdm_topk=2`）
- `head_type=residual_mlp`
- `pair_scorer=biaffine`
- `fusion_type=sum`
- `refiner_type=maskcnn`
- `span_encoder_type=none`

对应地，`train.py` 的相关可选参数已移除，不再保留多路实验开关。

## 2. 相对官方的主要改动模块

### 2.1 SDM（Self-adaptive Differentiation Module）

- 官方：`hard argmax + gumbel` 单一路径。
- 当前：固定为 `soft_topk_mix`（`topk=2`）单路径。

代码位置：
- `model/cnn_liabrary.py`
- `model/cnn.py`

### 2.2 BFM（Boundary Filtration Module）

- 官方：原始 `legacy` 过滤分支。
- 当前：固定为 `pyramid_gated`。

代码位置：
- `model/cnn.py`

### 2.3 最终打分头（Decoder Score Head）

- 官方：单层线性头。
- 当前：新增 `residual_mlp` 头（线性主分支 + 残差 MLP 分支），用于增强表达能力且保持训练稳定。

代码位置：
- `model/model.py`

### 2.4 Refiner 收敛

- 官方：`maskcnn`。
- 当前：固定为 `maskcnn`，`dilated` 分支已移除。

代码位置：
- `model/cnn.py`
- `model/model.py`

### 2.5 Span Semantic Encoder 收敛

- 官方：词级聚合后直接进入后续打分。
- 当前：固定为 `none`，`dwconv_res` 分支已移除。

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
  --cnn_depth 1 --n_head 4 --logit_drop 0.1 --seed 42 --n_layer 1
```

当前版本已固定最优主框架，不再支持 `dilated` 与 `dwconv_res` 等可选分支。

### 6.2 冲击 82 的建议训练命令（GENIA）

```bash
python train.py \
  -n 1 -d genia \
  --model_name /root/.cache/huggingface/hub/models--dmis-lab--biobert-v1.1/snapshots/551ca18efd7f052c8dfa0b01c94c2a8e68bc5488 \
  --lr 7e-6 --encoder_lr 2e-5 \
  --cnn_dim 200 --biaffine_size 200 --n_head 4 --cnn_depth 1 --n_layer 2 \
  -b 4 --accumulation_steps 2 --logit_drop 0.15 --loss_theta 1.5 \
  --load_model_dir /root/ner/_saved_models/2026-04-07-18_41_58_380108/model-epoch_1-batch_7512-f#f#test_81.54 \
  --adv_type fgm --adv_epsilon 1.0 \
  --ent_thres 0.5
```

训练后可做阈值扫描（先看 dev 最优，再取同阈值 test）：

```bash
python repro/sweep_genia_threshold.py \
  --model_name /root/.cache/huggingface/hub/models--dmis-lab--biobert-v1.1/snapshots/551ca18efd7f052c8dfa0b01c94c2a8e68bc5488 \
  --ckpt /root/ner/_saved_models/<your_run>/<your_ckpt_dir> \
  --thresholds 0.45:0.60:0.01 \
  --cnn_dim 200 --biaffine_size 200 --n_head 4 --cnn_depth 1 --n_layer 2 \
  --logit_drop 0.15 --loss_theta 1.5
```

## 7. 关键代码入口

- 训练参数入口：`train.py`
- 主模型与打分头：`model/model.py`
- SDM/BFM 与 Refiner：`model/cnn.py`
- SDM 掩码策略：`model/cnn_liabrary.py`

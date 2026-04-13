## 2026-04-13 04:02:27
- StageA FGM eps=0.02: log=log_20260413_032733, test@0.5=81.58, best_test_over_thres=81.64, test@best_dev=81.52
- StageB boundary_refine(scale=0.1, loss_w=0.2): log=log_20260413_035351, test@0.5=80.25 (drop)

## 2026-04-13 04:21:30
- StageA-variant fp16+FGM (b=8,acc=1): log=log_20260413_041029, test@0.5=68.68 (unstable collapse), artifacts removed.

## 2026-04-13 04:39:40
- StageA-continue from 81.58 (FGM, b6/acc2, lr=5e-6, enc_lr=1.5e-5): log=log_20260413_042323, test@0.5=81.29, best_test_over_thres=81.37 -> removed.

## 2026-04-13 04:58:20
- StageA seed sweep #1 (FGM, b5/acc2, seed=1): log=log_20260413_044100, test@0.5=81.03, best_test_over_thres=81.20 -> removed.

## 2026-04-13 05:39:10
- StageC PGD (eps=0.02, alpha=0.01, k=3, b4/acc2, seed=0): log=log_20260413_045941, test@0.5=81.57, best_test_over_thres=81.66, test@best_dev=81.54 (kept).

## 2026-04-13 05:58:05
- StageD PGD->FGM 2nd-stage fine-tune (lr=5e-6, enc_lr=1.5e-5, fgm_eps=0.015): log=log_20260413_054017, test@0.5=81.35 (drop) -> removed.

## 2026-04-13 06:13:05
- StageE boundary_refine-lite (scale=0.03, loss_w=0.0, no-adv): log=log_20260413_055948, test@0.5=81.50, best_test_over_thres=81.51, test@best_dev=81.46 -> removed.

## 2026-04-13 06:49:20
- StageF PGD seed sweep #2 (seed=1, eps=0.02, alpha=0.01, k=3): log=log_20260413_061344, test@0.5=81.44, best_test_over_thres=81.52 -> removed.

## 2026-04-13 06:51:10
- Cleanup: removed suboptimal 81.36 run (log_20260413_012522 and _saved_models/2026-04-13-01_25_04_855803).

## 2026-04-13 11:52:00
- StageG method1+2 gate (PGD + boundary_refine lite, 1epoch): log=log_20260413_111251, test@0.5=81.03, best_test_over_thres=81.08, gap_to_best81.66=0.58 -> dropped.

## 2026-04-13 12:50:40
- StageH method3 gate #1 (char-branch+gate, PGD, 1epoch): log=log_20260413_115917, test@0.5=81.30, best_test_over_thres=81.34, gap_to_best81.66=0.32 -> dropped.

## 2026-04-13 13:14:30
- StageI method3 gate #2 (char-branch+gate, FGM eps=0.01, b5/acc2, 1epoch): log=log_20260413_125248, test@0.5=81.04, best_test_over_thres=81.20, gap_to_best81.66=0.46 -> dropped.

## 2026-04-13 21:43:58
- StageJ adv-opt gate (PGD + adv_loss_weight=0.8 + adv_warmup_ratio=0.15 + random_start, 1epoch, b4/acc2): log=log_20260413_210215, test@0.5=81.34, best_test_over_thres=81.50 (thres=0.56), test@best_dev=81.33 -> below best81.66, dropped.

## 2026-04-13 22:41:42
- StageK SAD relation-bias gate (best PGD combo restored, 1epoch, b4/acc2, sad_relation_bias=true): log=log_20260413_220048, test@0.5=81.43, best_test_over_thres=81.55 (thres=0.55), test@best_dev=81.50 -> below best81.66, not promoted.

## 2026-04-13 23:38:42
- StageL SAD dynamic-depthwise gate (best PGD combo restored, 1epoch, b4/acc2, sad_dynamic_depthwise=true): log=log_20260413_225639, test@0.5=81.47, best_test_over_thres=81.59 (thres=0.53), test@best_dev=81.48 -> below best81.66, not promoted.

## 2026-04-14 00:32:18
- StageM SAD local-sparse-attn gate (best PGD combo restored, 1epoch, b4/acc2, sad_local_sparse_attn=true, sad_attn_topk=4): log=log_20260413_235057, test@0.5=81.63, best_test_over_thres=81.71 (thres=0.57), test@best_dev=81.58 -> new best over-threshold (+0.05 vs 81.66).

## 2026-04-14 02:59:42
- StageN local-sparse-attn 5epoch attempt (same PGD combo, b4/acc2) started at log=log_20260414_003822 and stopped by user to avoid overfitting drift.
- Snapshot at stop:
  - step=3756 (epoch1): dev=81.10, test=80.67
  - step=7512 (epoch2): dev=80.64, test=81.12
  - step=11268 (epoch3): dev=80.12, test=80.21
  - training was still running; no promoted checkpoint from this 5epoch run.

# Backup Summary (2026-04-14)

## Current Best (Single Model)
- Method: Local Sparse Attention SAD + PGD (warm-start from 81.54 ckpt)
- Run log: `logs/log_20260413_235057`
- Checkpoint: `_saved_models/2026-04-13-23_50_44_752068/model-epoch_1-batch_3756-f#f#test_81.63/fastnlp_model.pkl.tar`
- `test@0.5`: **81.63**
- `best_test_over_thres`: **81.71** at threshold **0.57**
- Sweep file: `repro/tmp_sad_local_sparse_235057_sweep.json`

## Key Reference Files
- Experiment journal: `repro/experiment_journal.md`
- Threshold sweep script: `repro/sweep_genia_threshold.py`
- Imported reproduction script: `repro/import_20260413/train_20260407_imported.py`

## Method Gate Results (Recent)
- StageK (SAD relation-bias): best over thres 81.55
- StageL (SAD dynamic-depthwise): best over thres 81.59
- StageM (SAD local-sparse-attn): best over thres **81.71** (current best)

## Note
- A later 5-epoch run (`logs/log_20260414_003822`) was stopped and not promoted.

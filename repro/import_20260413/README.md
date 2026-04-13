# Imported Assets From `/root/ner_repro_8154_20260413`

Date merged: 2026-04-13

## Merged into `/root/ner`

1. 81.54 log directory:
   - `/root/ner/logs/log_20260407_184212`

2. 81.54 checkpoint directory:
   - `/root/ner/_saved_models/2026-04-07-18_41_58_380108/model-epoch_1-batch_7512-f#f#test_81.54`

3. Biobert HF cache (filled into global cache):
   - `/root/.cache/huggingface/hub/models--dmis-lab--biobert-v1.1`
   - snapshot `551ca18efd7f052c8dfa0b01c94c2a8e68bc5488` now contains `pytorch_model.bin`

## Kept as reference only (not overriding current code)

- `/root/ner/repro/import_20260413/train_20260407_imported.py`
- `/root/ner/repro/import_20260413/ner_pipe_20260407_imported.py`
- `/root/ner/repro/import_20260413/hyper_81_54.log`
- `/root/ner/repro/import_20260413/metric_81_54.log`

## Notes

- GENIA jsonlines in imported package are byte-identical to current `/root/ner/preprocess/outputs/genia/*`.
- Imported `hyper.log` indicates the 81.54 run used:
  - `seed=42`, `batch_size=2`, `accumulation_steps=4`, `loss_theta=1.5`
  - `bfm_type=pyramid_gated`, `sdm_mask_type=soft_topk_mix`, `head_type=residual_mlp`
  - `cnn_dim=200`
  - `load_model_dir=/mnt/workspace/ner/_saved_models/2026-04-06-13_40_54_453643/model_epoch_7_batch_52584_f#f#test_81.2` (warm-start source not included in imported folder)

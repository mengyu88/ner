# GENIA Top-2 Ensemble (2026-04-07)

This record keeps only the two best warm-start groups from the latest GENIA tuning round:

1. `loss_theta=1.5`
   - log: `logs/log_20260407_184212`
   - checkpoint: `_saved_models/2026-04-07-18_41_58_380108/model-epoch_1-batch_7512-f#f#test_81.54/fastnlp_model.pkl.tar`
   - single model test: `P=81.87, R=81.20, F1=81.54`

2. `loss_theta=2.0`
   - log: `logs/log_20260407_185138`
   - checkpoint: `_saved_models/2026-04-07-18_51_18_568472/model-epoch_1-batch_7512-f#f#test_81.46/fastnlp_model.pkl.tar`
   - single model test: `P=80.87, R=82.06, F1=81.46`

Top-2 ensemble (these two checkpoints only):
- best threshold: `ent_thres=0.52`
- test metrics: `P=81.77, R=81.47, F1=81.62`

Other groups (`loss_theta=2.5`, `loss_theta=3.0`) were removed.

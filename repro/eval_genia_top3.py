#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from fastNLP import SortedSampler, cache_results, prepare_torch_dataloader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.ner_pipe import SpanNerPipe
from data.padder import Torch3DMatrixPadder
from model.metrics import NERMetric
from model.model import CNNNer


BEST_FRAMEWORK = {
    'size_feature_type': 'embed',
    'word_pooling': 'max',
    'pool_gate_type': 'scalar',
    'refiner_type': 'maskcnn',
    'span_encoder_type': 'none',
    'pair_scorer': 'biaffine',
    'mlp_type': 'mlp',
    'fusion_type': 'sum',
    'head_type': 'residual_mlp',
    'bfm_type': 'pyramid_gated',
    'sdm_mask_type': 'soft_topk_mix',
    'sdm_topk': 2,
}


def densify(x):
    return x.todense().astype(np.float32)


@cache_results("caches/ner_caches.pkl", _refresh=False)
def get_data(model_name):
    pipe = SpanNerPipe(model_name=model_name)
    data_bundle = pipe.process_from_file("preprocess/outputs/genia")
    return data_bundle, pipe.matrix_segs


def build_test_loader(data_bundle, matrix_segs, batch_size):
    test_ds = data_bundle.get_dataset("test")
    test_ds.set_pad(
        "matrix",
        pad_fn=Torch3DMatrixPadder(
            pad_val=test_ds.collator.input_fields["matrix"]["pad_val"],
            num_class=matrix_segs["ent"],
            batch_size=batch_size,
        ),
    )
    return prepare_torch_dataloader(
        test_ds,
        batch_size=batch_size,
        num_workers=2,
        sampler=SortedSampler(test_ds, "input_ids"),
        pin_memory=True,
        shuffle=False,
    )


def build_model(
    model_name,
    loss_theta,
    cnn_dim,
    biaffine_size,
    n_head,
    cnn_depth,
    n_layer,
    logit_drop,
):
    return CNNNer(
        model_name,
        num_ner_tag=5,
        cnn_dim=cnn_dim,
        biaffine_size=biaffine_size,
        size_embed_dim=25,
        logit_drop=logit_drop,
        n_layer=n_layer,
        kernel_size=3,
        n_head=n_head,
        cnn_depth=cnn_depth,
        separateness_rate=0.05,
        theta=1.0,
        loss_theta=loss_theta,
        **BEST_FRAMEWORK,
    )


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def parse_thresholds(raw: str) -> List[float]:
    if "," in raw:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
        return [round(v, 4) for v in vals]
    start, end, step = [float(x.strip()) for x in raw.split(":")]
    vals = np.arange(start, end + 1e-9, step).tolist()
    return [round(v, 4) for v in vals]


def collect(metrics_dict, thresholds):
    table = []
    for t in thresholds:
        m = metrics_dict[t].get_metric()
        table.append({"thres": t, "f1": m["f"], "rec": m["rec"], "pre": m["pre"]})
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--ckpt1", required=True)
    parser.add_argument("--ckpt2", required=True)
    parser.add_argument("--ckpt3", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--thresholds", type=str, default="0.45:0.62:0.01")
    parser.add_argument("--cnn_dim", type=int, default=200)
    parser.add_argument("--biaffine_size", type=int, default=200)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--cnn_depth", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--logit_drop", type=float, default=0.15)
    parser.add_argument("--loss_theta1", type=float, default=1.5)
    parser.add_argument("--loss_theta2", type=float, default=1.5)
    parser.add_argument("--loss_theta3", type=float, default=1.5)
    args = parser.parse_args()

    assert os.path.isdir(args.model_name), f"model_name is not a directory: {args.model_name}"
    assert os.path.isfile(args.ckpt1), f"checkpoint not found: {args.ckpt1}"
    assert os.path.isfile(args.ckpt2), f"checkpoint not found: {args.ckpt2}"
    assert os.path.isfile(args.ckpt3), f"checkpoint not found: {args.ckpt3}"

    thresholds = parse_thresholds(args.thresholds)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_bundle, matrix_segs = get_data(args.model_name)
    data_bundle.apply_field(densify, field_name="matrix", new_field_name="matrix", progress_bar="Densify")
    test_loader = build_test_loader(data_bundle, matrix_segs, batch_size=args.batch_size)

    m1 = build_model(args.model_name, args.loss_theta1, args.cnn_dim, args.biaffine_size, args.n_head,
                     args.cnn_depth, args.n_layer, args.logit_drop).to(device)
    m2 = build_model(args.model_name, args.loss_theta2, args.cnn_dim, args.biaffine_size, args.n_head,
                     args.cnn_depth, args.n_layer, args.logit_drop).to(device)
    m3 = build_model(args.model_name, args.loss_theta3, args.cnn_dim, args.biaffine_size, args.n_head,
                     args.cnn_depth, args.n_layer, args.logit_drop).to(device)

    m1.load_state_dict(torch.load(args.ckpt1, map_location="cpu"))
    m2.load_state_dict(torch.load(args.ckpt2, map_location="cpu"))
    m3.load_state_dict(torch.load(args.ckpt3, map_location="cpu"))
    m1.eval()
    m2.eval()
    m3.eval()

    single1 = {t: NERMetric(matrix_segs=matrix_segs, ent_thres=t, allow_nested=True) for t in thresholds}
    single2 = {t: NERMetric(matrix_segs=matrix_segs, ent_thres=t, allow_nested=True) for t in thresholds}
    single3 = {t: NERMetric(matrix_segs=matrix_segs, ent_thres=t, allow_nested=True) for t in thresholds}
    ensemble = {t: NERMetric(matrix_segs=matrix_segs, ent_thres=t, allow_nested=True) for t in thresholds}

    with torch.no_grad():
        for batch in test_loader:
            batch = move_to_device(batch, device)
            inp = {
                "input_ids": batch["input_ids"],
                "bpe_len": batch["bpe_len"],
                "indexes": batch["indexes"],
                "matrix": batch["matrix"],
                "raw_words": batch["raw_words"],
            }
            out1 = m1(**inp)["scores"]
            out2 = m2(**inp)["scores"]
            out3 = m3(**inp)["scores"]
            out_e = (out1 + out2 + out3) / 3.0
            for t in thresholds:
                single1[t].update(ent_target=batch["ent_target"], scores=out1, word_len=batch["word_len"])
                single2[t].update(ent_target=batch["ent_target"], scores=out2, word_len=batch["word_len"])
                single3[t].update(ent_target=batch["ent_target"], scores=out3, word_len=batch["word_len"])
                ensemble[t].update(ent_target=batch["ent_target"], scores=out_e, word_len=batch["word_len"])

    r1 = collect(single1, thresholds)
    r2 = collect(single2, thresholds)
    r3 = collect(single3, thresholds)
    re = collect(ensemble, thresholds)

    best1 = max(r1, key=lambda x: x["f1"])
    best2 = max(r2, key=lambda x: x["f1"])
    best3 = max(r3, key=lambda x: x["f1"])
    beste = max(re, key=lambda x: x["f1"])

    print(json.dumps({"single1": r1, "single2": r2, "single3": r3, "ensemble": re}, ensure_ascii=False, indent=2))
    print("BEST_SINGLE1", json.dumps(best1, ensure_ascii=False))
    print("BEST_SINGLE2", json.dumps(best2, ensure_ascii=False))
    print("BEST_SINGLE3", json.dumps(best3, ensure_ascii=False))
    print("BEST_ENSEMBLE", json.dumps(beste, ensure_ascii=False))


if __name__ == "__main__":
    main()

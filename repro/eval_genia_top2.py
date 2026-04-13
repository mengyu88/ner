#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from fastNLP import SortedSampler, cache_results, prepare_torch_dataloader

# Make `data/` and `model/` importable when running as a script.
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
    boundary_refine=False,
    boundary_scale=0.1,
    boundary_loss_weight=0.2,
    char_branch=False,
    char_vocab_size=2048,
    char_dim=32,
    char_max_len=16,
    char_dropout=0.1,
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
        boundary_refine=boundary_refine,
        boundary_scale=boundary_scale,
        boundary_loss_weight=boundary_loss_weight,
        char_branch=char_branch,
        char_vocab_size=char_vocab_size,
        char_dim=char_dim,
        char_max_len=char_max_len,
        char_dropout=char_dropout,
        **BEST_FRAMEWORK,
    )


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def parse_thresholds(raw: str) -> List[float]:
    if "," in raw:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
        return [round(v, 4) for v in vals]
    start, end, step = [float(x.strip()) for x in raw.split(":")]
    vals = np.arange(start, end + 1e-9, step).tolist()
    return [round(v, 4) for v in vals]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Local encoder dir")
    parser.add_argument("--ckpt1", required=True)
    parser.add_argument("--ckpt2", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.48:0.56:0.01",
        help="Either start:end:step or comma list, e.g. 0.48:0.56:0.01",
    )
    parser.add_argument("--cnn_dim", type=int, default=200)
    parser.add_argument("--biaffine_size", type=int, default=200)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--cnn_depth", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--logit_drop", type=float, default=0.15)
    parser.add_argument("--loss_theta1", type=float, default=1.5)
    parser.add_argument("--loss_theta2", type=float, default=2.0)
    parser.add_argument("--boundary_refine", action='store_true')
    parser.add_argument("--boundary_scale", type=float, default=0.1)
    parser.add_argument("--boundary_loss_weight", type=float, default=0.2)
    parser.add_argument("--char_branch", action='store_true')
    parser.add_argument("--char_vocab_size", type=int, default=2048)
    parser.add_argument("--char_dim", type=int, default=32)
    parser.add_argument("--char_max_len", type=int, default=16)
    parser.add_argument("--char_dropout", type=float, default=0.1)
    args = parser.parse_args()

    assert os.path.isdir(args.model_name), f"model_name is not a directory: {args.model_name}"
    assert os.path.isfile(args.ckpt1), f"checkpoint not found: {args.ckpt1}"
    assert os.path.isfile(args.ckpt2), f"checkpoint not found: {args.ckpt2}"

    thresholds = parse_thresholds(args.thresholds)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_bundle, matrix_segs = get_data(args.model_name)
    data_bundle.apply_field(densify, field_name="matrix", new_field_name="matrix", progress_bar="Densify")
    test_loader = build_test_loader(data_bundle, matrix_segs, batch_size=args.batch_size)

    m1 = build_model(
        args.model_name,
        loss_theta=args.loss_theta1,
        cnn_dim=args.cnn_dim,
        biaffine_size=args.biaffine_size,
        n_head=args.n_head,
        cnn_depth=args.cnn_depth,
        n_layer=args.n_layer,
        logit_drop=args.logit_drop,
        boundary_refine=args.boundary_refine,
        boundary_scale=args.boundary_scale,
        boundary_loss_weight=args.boundary_loss_weight,
        char_branch=args.char_branch,
        char_vocab_size=args.char_vocab_size,
        char_dim=args.char_dim,
        char_max_len=args.char_max_len,
        char_dropout=args.char_dropout,
    ).to(device)
    m2 = build_model(
        args.model_name,
        loss_theta=args.loss_theta2,
        cnn_dim=args.cnn_dim,
        biaffine_size=args.biaffine_size,
        n_head=args.n_head,
        cnn_depth=args.cnn_depth,
        n_layer=args.n_layer,
        logit_drop=args.logit_drop,
        boundary_refine=args.boundary_refine,
        boundary_scale=args.boundary_scale,
        boundary_loss_weight=args.boundary_loss_weight,
        char_branch=args.char_branch,
        char_vocab_size=args.char_vocab_size,
        char_dim=args.char_dim,
        char_max_len=args.char_max_len,
        char_dropout=args.char_dropout,
    ).to(device)
    m1.load_state_dict(torch.load(args.ckpt1, map_location="cpu"))
    m2.load_state_dict(torch.load(args.ckpt2, map_location="cpu"))
    m1.eval()
    m2.eval()

    single1 = {t: NERMetric(matrix_segs=matrix_segs, ent_thres=t, allow_nested=True) for t in thresholds}
    single2 = {t: NERMetric(matrix_segs=matrix_segs, ent_thres=t, allow_nested=True) for t in thresholds}
    ens = {t: NERMetric(matrix_segs=matrix_segs, ent_thres=t, allow_nested=True) for t in thresholds}

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
            out_ens = (out1 + out2) / 2
            for t in thresholds:
                single1[t].update(ent_target=batch["ent_target"], scores=out1, word_len=batch["word_len"])
                single2[t].update(ent_target=batch["ent_target"], scores=out2, word_len=batch["word_len"])
                ens[t].update(ent_target=batch["ent_target"], scores=out_ens, word_len=batch["word_len"])

    def collect(metrics_dict):
        table = []
        for t in thresholds:
            m = metrics_dict[t].get_metric()
            row = {"thres": t, "f1": m["f"], "rec": m["rec"], "pre": m["pre"]}
            table.append(row)
        return table

    r1 = collect(single1)
    r2 = collect(single2)
    re = collect(ens)

    best1 = max(r1, key=lambda x: x["f1"])
    best2 = max(r2, key=lambda x: x["f1"])
    beste = max(re, key=lambda x: x["f1"])

    print(json.dumps({"single1": r1, "single2": r2, "ensemble": re}, ensure_ascii=False, indent=2))
    print("BEST_SINGLE1", json.dumps(best1, ensure_ascii=False))
    print("BEST_SINGLE2", json.dumps(best2, ensure_ascii=False))
    print("BEST_ENSEMBLE", json.dumps(beste, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

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


@cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(model_name: str):
    pipe = SpanNerPipe(model_name=model_name)
    data_bundle = pipe.process_from_file('preprocess/outputs/genia')
    return data_bundle, pipe.matrix_segs


def build_loader(ds, matrix_segs, batch_size):
    ds.set_pad(
        'matrix',
        pad_fn=Torch3DMatrixPadder(
            pad_val=ds.collator.input_fields['matrix']['pad_val'],
            num_class=matrix_segs['ent'],
            batch_size=batch_size,
        ),
    )
    return prepare_torch_dataloader(
        ds,
        batch_size=batch_size,
        num_workers=2,
        sampler=SortedSampler(ds, 'input_ids'),
        pin_memory=True,
        shuffle=False,
    )


def parse_thresholds(raw: str) -> List[float]:
    if ',' in raw:
        vals = [float(x.strip()) for x in raw.split(',') if x.strip()]
        return [round(v, 4) for v in vals]
    start, end, step = [float(x.strip()) for x in raw.split(':')]
    vals = np.arange(start, end + 1e-9, step).tolist()
    return [round(v, 4) for v in vals]


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def resolve_ckpt_path(path_or_dir: str) -> str:
    if os.path.isfile(path_or_dir):
        return path_or_dir
    candidate = os.path.join(path_or_dir, 'fastnlp_model.pkl.tar')
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f'Cannot find checkpoint file from: {path_or_dir}')


def build_model(args, num_ner_tag):
    return CNNNer(
        model_name=args.model_name,
        num_ner_tag=num_ner_tag,
        cnn_dim=args.cnn_dim,
        biaffine_size=args.biaffine_size,
        size_embed_dim=25,
        logit_drop=args.logit_drop,
        n_layer=args.n_layer,
        kernel_size=3,
        n_head=args.n_head,
        cnn_depth=args.cnn_depth,
        separateness_rate=0.05,
        theta=1.0,
        loss_theta=args.loss_theta,
        boundary_refine=args.boundary_refine,
        boundary_scale=args.boundary_scale,
        boundary_loss_weight=args.boundary_loss_weight,
        char_branch=args.char_branch,
        char_vocab_size=args.char_vocab_size,
        char_dim=args.char_dim,
        char_max_len=args.char_max_len,
        char_dropout=args.char_dropout,
        sad_relation_bias=args.sad_relation_bias,
        sad_dynamic_depthwise=args.sad_dynamic_depthwise,
        sad_local_sparse_attn=args.sad_local_sparse_attn,
        sad_attn_topk=args.sad_attn_topk,
        **BEST_FRAMEWORK,
    )


def evaluate_split(model, data_loader, matrix_segs, thresholds, device):
    metrics = {t: NERMetric(matrix_segs=matrix_segs, ent_thres=t, allow_nested=True) for t in thresholds}
    with torch.no_grad():
        for batch in data_loader:
            batch = move_to_device(batch, device)
            outputs = model(
                input_ids=batch['input_ids'],
                bpe_len=batch['bpe_len'],
                indexes=batch['indexes'],
                matrix=batch['matrix'],
                raw_words=batch['raw_words'],
            )
            scores = outputs['scores']
            for t in thresholds:
                metrics[t].update(ent_target=batch['ent_target'], scores=scores, word_len=batch['word_len'])
    rows = []
    for t in thresholds:
        m = metrics[t].get_metric()
        rows.append({'thres': t, 'f1': m['f'], 'rec': m['rec'], 'pre': m['pre']})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, help='Encoder path or HF name')
    parser.add_argument('--ckpt', required=True, help='Checkpoint file or folder containing fastnlp_model.pkl.tar')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--thresholds', type=str, default='0.45:0.60:0.01')

    parser.add_argument('--cnn_dim', type=int, default=200)
    parser.add_argument('--biaffine_size', type=int, default=200)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--cnn_depth', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--logit_drop', type=float, default=0.15)
    parser.add_argument('--loss_theta', type=float, default=1.5)
    parser.add_argument('--boundary_refine', action='store_true')
    parser.add_argument('--boundary_scale', type=float, default=0.1)
    parser.add_argument('--boundary_loss_weight', type=float, default=0.2)
    parser.add_argument('--char_branch', action='store_true')
    parser.add_argument('--char_vocab_size', type=int, default=2048)
    parser.add_argument('--char_dim', type=int, default=32)
    parser.add_argument('--char_max_len', type=int, default=16)
    parser.add_argument('--char_dropout', type=float, default=0.1)
    parser.add_argument('--sad_relation_bias', action='store_true')
    parser.add_argument('--sad_dynamic_depthwise', action='store_true')
    parser.add_argument('--sad_local_sparse_attn', action='store_true')
    parser.add_argument('--sad_attn_topk', type=int, default=4)
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    ckpt_path = resolve_ckpt_path(args.ckpt)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_bundle, matrix_segs = get_data(args.model_name)
    data_bundle.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')

    dev_loader = build_loader(data_bundle.get_dataset('dev'), matrix_segs, args.batch_size)
    test_loader = build_loader(data_bundle.get_dataset('test'), matrix_segs, args.batch_size)

    model = build_model(args, num_ner_tag=matrix_segs['ent']).to(device)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dev_rows = evaluate_split(model, dev_loader, matrix_segs, thresholds, device)
    test_rows = evaluate_split(model, test_loader, matrix_segs, thresholds, device)

    best_dev = max(dev_rows, key=lambda x: x['f1'])
    test_map = {r['thres']: r for r in test_rows}
    test_at_best_dev = test_map[best_dev['thres']]

    payload = {
        'dev': dev_rows,
        'test': test_rows,
        'best_dev': best_dev,
        'test_at_best_dev_thres': test_at_best_dev,
        'ckpt': ckpt_path,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

import json
import os
import warnings
import argparse
# from torchstat import stat
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import random
# from thop import profile
from fastNLP import cache_results, prepare_torch_dataloader
from fastNLP import print
from fastNLP import Trainer
from fastNLP import FitlogCallback, CheckpointCallback, TorchGradClipCallback
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP import TorchWarmupCallback
import fitlog

# Compatibility for legacy fitlog versions on NumPy>=2.
for _alias, _type in (('str', str), ('int', int), ('float', float), ('bool', bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _type)

# fitlog.debug()

from model.model import CNNNer
from model.metrics import NERMetric
from data.ner_pipe import SpanNerPipe
from data.padder import Torch3DMatrixPadder
from callbacks import AdversarialTrainingCallback

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--encoder_lr', default=2e-5, type=float)
parser.add_argument('-b', '--batch_size', default=24, type=int)
parser.add_argument('-n', '--n_epochs', default=50, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='ace2004', type=str) # ace2005 genia ace2004
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--ent_thres', default=0.5, type=float)
parser.add_argument('--cnn_depth', default=1, type=int)
parser.add_argument('--cnn_dim', default=120, type=int)
parser.add_argument('--num', default=1, type=int)
parser.add_argument('--logit_drop', default=0, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--n_head', default=5, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--n_layer', default=1, type=int)
parser.add_argument('--accumulation_steps', default=1, type=int)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--separateness_rate', default=5, type=int)
parser.add_argument('--theta', default=1, type=float)
parser.add_argument('--loss_theta', default=1, type=float)
parser.add_argument('--load_model_dir', default=None, type=str,
                    help='Folder that contains fastnlp_model.pkl.tar to warm-start training.')
parser.add_argument('--adv_type', default='none', choices=['none', 'fgm', 'pgd'], type=str)
parser.add_argument('--adv_epsilon', default=1.0, type=float)
parser.add_argument('--adv_alpha', default=0.3, type=float)
parser.add_argument('--adv_k', default=3, type=int)
parser.add_argument('--adv_emb_name', default='word_embeddings', type=str)
parser.add_argument('--adv_loss_weight', default=1.0, type=float)
parser.add_argument('--adv_warmup_ratio', default=0.0, type=float)
parser.add_argument('--adv_every_n_steps', default=1, type=int)
parser.add_argument('--adv_random_start', action='store_true')
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--boundary_refine', action='store_true')
parser.add_argument('--boundary_scale', default=0.1, type=float)
parser.add_argument('--boundary_loss_weight', default=0.2, type=float)
parser.add_argument('--char_branch', action='store_true')
parser.add_argument('--char_vocab_size', default=2048, type=int)
parser.add_argument('--char_dim', default=32, type=int)
parser.add_argument('--char_max_len', default=16, type=int)
parser.add_argument('--char_dropout', default=0.1, type=float)
parser.add_argument('--class_pos_weights', default='', type=str,
                    help='Comma-separated class positive weights, e.g. 1.2,1.0,1.4,1.0,1.1')
parser.add_argument('--auto_class_pos_weight', action='store_true')
parser.add_argument('--class_weight_power', default=0.5, type=float)
parser.add_argument('--class_weight_min', default=0.8, type=float)
parser.add_argument('--class_weight_max', default=1.8, type=float)
parser.add_argument('--short_span_pos_boost', default=1.0, type=float)
parser.add_argument('--short_span_neg_boost', default=1.0, type=float)
parser.add_argument('--short_span_max_len', default=2, type=int)
parser.add_argument('--sad_relation_bias', action='store_true')
parser.add_argument('--sad_dynamic_depthwise', action='store_true')
parser.add_argument('--sad_local_sparse_attn', action='store_true')
parser.add_argument('--sdm_topk', default=2, type=int)
parser.add_argument('--sad_attn_topk', default=4, type=int)
parser.add_argument('--sad_attn_heads', default=1, type=int)
parser.add_argument('--sad_relation_rich_bias', action='store_true')
parser.add_argument('--sad_dual_path_fusion', action='store_true')
parser.add_argument('--sad_dual_path_gate_map', action='store_true')

# Keep only the current best-performing framework path.
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
}


def default_model_name(dataset_name):
    if 'genia' in dataset_name:
        return 'dmis-lab/biobert-v1.1'
    if dataset_name == 'weibo':
        return 'bert-base-chinese'
    if dataset_name == 'conll03':
        return 'bert-large-cased'
    if dataset_name in ('ace2004', 'ace2005'):
        return 'roberta-base'
    raise RuntimeError(f'Unsupported dataset_name: {dataset_name}')


def resolve_model_name(dataset_name, model_name):
    model_name = model_name or default_model_name(dataset_name)
    if os.path.isdir(model_name):
        return model_name

    use_modelscope = os.environ.get('USE_MODELSCOPE', '0') == '1' or model_name.startswith('ms://')
    if not use_modelscope:
        return model_name

    model_id = model_name[5:] if model_name.startswith('ms://') else model_name
    modelscope_candidates = {
        'roberta-base': ['AI-ModelScope/roberta-base', 'roberta-base'],
        'bert-large-cased': ['AI-ModelScope/bert-large-cased', 'bert-large-cased'],
        'bert-base-chinese': ['AI-ModelScope/bert-base-chinese', 'bert-base-chinese'],
        'dmis-lab/biobert-v1.1': ['dmis-lab/biobert-v1.1', 'AI-ModelScope/biobert-v1.1'],
    }
    candidate_ids = modelscope_candidates.get(model_id, [model_id])
    from modelscope.hub.snapshot_download import snapshot_download
    cache_dir = os.environ.get('MODELSCOPE_CACHE', None)
    last_error = None
    for candidate in candidate_ids:
        try:
            model_dir = snapshot_download(candidate, cache_dir=cache_dir)
            print(f'Loaded model from ModelScope: {candidate} -> {model_dir}')
            return model_dir
        except Exception as e:
            last_error = e
    raise RuntimeError(
        f'Failed to download model via ModelScope, candidates={candidate_ids}. '
        f'You can pass --model_name <local_path>, or use HF mirror via HF_ENDPOINT. '
        f'Last error: {last_error}'
    )


args = parser.parse_args()
dataset_name = args.dataset_name
model_name = resolve_model_name(dataset_name, args.model_name)
args.model_name = model_name
n_head = args.n_head
######hyper
non_ptm_lr_ratio = 100
schedule = 'linear'
weight_decay = 1e-4
size_embed_dim = 25
ent_thres = args.ent_thres
kernel_size = 3
######hyper

fitlog.set_log_dir('logs/')

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
def seed_torch(seed=43):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False


seed = fitlog.set_rng_seed(rng_seed=args.seed)
seed_torch(args.seed)
os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
fitlog.add_hyper(args)
framework_hyper = dict(BEST_FRAMEWORK)
framework_hyper['sdm_topk'] = args.sdm_topk
fitlog.add_hyper(framework_hyper, name='best_framework')
fitlog.add_hyper_in_file(__file__)

@cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name):
    # 以下是我们自己的数据
    if dataset_name == 'ace2004':
        paths = 'preprocess/outputs/ace2004'
    elif dataset_name == 'ace2005':
        paths = 'preprocess/outputs/ace2005'
    elif dataset_name == 'genia':
        paths = 'preprocess/outputs/genia'
    elif dataset_name == 'weibo':
        paths = 'preprocess/outputs/weibo'
    elif dataset_name == 'conll03':
        paths = 'preprocess/outputs/conll03'
    else:
        raise RuntimeError("Does not support.")
    pipe = SpanNerPipe(model_name=model_name)
    dl = pipe.process_from_file(paths)

    return dl, pipe.matrix_segs


dl, matrix_segs = get_data(dataset_name, model_name)


def densify(x):
    x = x.todense().astype(np.float32)
    return x


dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')

print(dl)
label2idx = getattr(dl, 'ner_vocab') if hasattr(dl, 'ner_vocab') else getattr(dl, 'label2idx')
print(f"{len(label2idx)} labels: {label2idx}, matrix_segs:{matrix_segs}")


def _parse_class_pos_weights(raw, num_class):
    raw = (raw or '').strip()
    if not raw:
        return None
    vals = [float(x.strip()) for x in raw.split(',') if x.strip()]
    if len(vals) != num_class:
        raise ValueError(f'class_pos_weights length {len(vals)} != num_class {num_class}')
    return vals


def _compute_auto_class_pos_weights(train_ds, num_class, power, min_w, max_w):
    pos_counts = np.zeros(num_class, dtype=np.float64)
    for ins in train_ds:
        matrix = ins['matrix']
        if hasattr(matrix, 'todense'):
            matrix = matrix.todense()
        matrix = np.asarray(matrix)
        pos_counts += (matrix > 0.5).reshape(-1, num_class).sum(axis=0)
    mean_pos = pos_counts.mean() + 1e-12
    weights = np.power(mean_pos / (pos_counts + 1e-12), power)
    weights = weights / (weights.mean() + 1e-12)
    weights = np.clip(weights, min_w, max_w)
    return weights.tolist(), pos_counts.tolist()


class_pos_weights = _parse_class_pos_weights(args.class_pos_weights, matrix_segs['ent'])
if class_pos_weights is None and args.auto_class_pos_weight:
    class_pos_weights, pos_counts = _compute_auto_class_pos_weights(
        dl.get_dataset('train'),
        matrix_segs['ent'],
        args.class_weight_power,
        args.class_weight_min,
        args.class_weight_max,
    )
    print(f'Auto class_pos_weights={class_pos_weights}, pos_counts={pos_counts}')
if class_pos_weights is not None:
    fitlog.add_other(value=class_pos_weights, name='class_pos_weights')

dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=matrix_segs['ent'],
                                                    batch_size=args.batch_size))

    if name == 'train':
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                       batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                          batch_size=args.batch_size,
                                                                          num_batch_per_bucket=30),
                                       pin_memory=True, shuffle=True)

    else:
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                       sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
    dls[name] = _dl

model = CNNNer(model_name, num_ner_tag=matrix_segs['ent'], cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
               size_embed_dim=size_embed_dim, logit_drop=args.logit_drop,n_layer=args.n_layer,
               kernel_size=kernel_size, n_head=n_head, cnn_depth=args.cnn_depth,
               separateness_rate=args.separateness_rate/100, theta=args.theta,
               loss_theta=args.loss_theta,
               boundary_refine=args.boundary_refine,
               boundary_scale=args.boundary_scale,
               boundary_loss_weight=args.boundary_loss_weight,
               char_branch=args.char_branch,
               char_vocab_size=args.char_vocab_size,
               char_dim=args.char_dim,
               char_max_len=args.char_max_len,
               char_dropout=args.char_dropout,
               class_pos_weights=class_pos_weights,
               short_span_pos_boost=args.short_span_pos_boost,
               short_span_neg_boost=args.short_span_neg_boost,
               short_span_max_len=args.short_span_max_len,
               sad_relation_bias=args.sad_relation_bias,
               sad_dynamic_depthwise=args.sad_dynamic_depthwise,
               sad_local_sparse_attn=args.sad_local_sparse_attn,
               sdm_topk=args.sdm_topk,
               sad_attn_topk=args.sad_attn_topk,
               sad_attn_heads=args.sad_attn_heads,
               sad_relation_rich_bias=args.sad_relation_rich_bias,
               sad_dual_path_fusion=args.sad_dual_path_fusion,
               sad_dual_path_gate_map=args.sad_dual_path_gate_map,
               **BEST_FRAMEWORK)

# optimizer
parameters = []
ln_params = []
non_ln_params = []
non_pretrain_params = []
non_pretrain_ln_params = []

import collections

counter = collections.Counter()
for name, param in model.named_parameters():
    counter[name.split('.')[0]] += torch.numel(param)
print(counter)
print("Total param ", sum(counter.values()))
fitlog.add_to_line(json.dumps(counter, indent=2))
fitlog.add_other(value=sum(counter.values()), name='total_param')
for name, param in model.named_parameters():
    name = name.lower()
    if param.requires_grad is False:
        continue
    if 'pretrain_model' in name:
        if 'norm' in name or 'bias' in name:
            ln_params.append(param)
        else:
            non_ln_params.append(param)
    else:
        if 'norm' in name or 'bias' in name:
            non_pretrain_ln_params.append(param)
        else:      
            non_pretrain_params.append(param)
optimizer = torch.optim.AdamW([{'params': non_ln_params, 'lr': args.encoder_lr, 'weight_decay': weight_decay},
                               {'params': ln_params, 'lr': args.encoder_lr, 'weight_decay': 0},
                               {'params': non_pretrain_ln_params, 'lr': args.lr * 
                                non_ptm_lr_ratio, 'weight_decay': 0},
                               {'params': non_pretrain_params, 'lr': args.lr * non_ptm_lr_ratio,
                                'weight_decay': weight_decay}])
# callbacks
callbacks = []
callbacks.append(FitlogCallback(log_loss_every=20))
callbacks.append(CheckpointCallback(monitor='f#f#test',save_evaluate_results=True, folder='_saved_models', topk=3))
callbacks.append(TorchGradClipCallback(clip_value=5))
callbacks.append(TorchWarmupCallback(warmup=args.warmup, schedule=schedule))
if args.adv_type != 'none':
    callbacks.append(
        AdversarialTrainingCallback(
            adv_type=args.adv_type,
            epsilon=args.adv_epsilon,
            alpha=args.adv_alpha,
            k=args.adv_k,
            emb_name=args.adv_emb_name,
            adv_loss_weight=args.adv_loss_weight,
            adv_warmup_ratio=args.adv_warmup_ratio,
            adv_every_n_steps=args.adv_every_n_steps,
            pgd_random_start=args.adv_random_start,
        )
    )
    print(
        f'Enable adversarial training: type={args.adv_type}, epsilon={args.adv_epsilon}, '
        f'alpha={args.adv_alpha}, k={args.adv_k}, emb_name={args.adv_emb_name}, '
        f'loss_weight={args.adv_loss_weight}, warmup_ratio={args.adv_warmup_ratio}, '
        f'every_n_steps={args.adv_every_n_steps}, random_start={args.adv_random_start}'
    )
train_dls = {}
evaluate_dls = {}

if 'dev' in dls:
    evaluate_dls['dev'] = dls['dev']
if 'test' in dls:
    evaluate_dls['test'] = dls['test']
allow_nested = True
metrics = {'f': NERMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, allow_nested=allow_nested)}

trainer = Trainer(model=model,
                  driver='torch',
                  train_dataloader=dls.get('train'),
                  evaluate_dataloaders=evaluate_dls,
                  optimizers=optimizer,
                  callbacks=callbacks,
                  overfit_batches=0,
                  device=0 if torch.cuda.is_available() else 'cpu',
                  n_epochs=args.n_epochs,
                  metrics=metrics,
                  monitor='f#f#dev',
                  evaluate_every=-1,
                  evaluate_use_dist_sampler=True,
                  accumulation_steps=args.accumulation_steps,
                  fp16=args.fp16,
                  progress_bar='rich')

if args.load_model_dir:
    print(f'Load model weights from: {args.load_model_dir}')
    trainer.load_model(args.load_model_dir, only_state_dict=True, strict=False)

trainer.run(num_train_batch_per_epoch=-1, num_eval_batch_per_dl=-1, num_eval_sanity_batch=1)
fitlog.finish()

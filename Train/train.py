"""PCVRHyFormer 的训练入口。

用法：
    python train.py [--num_epochs 10] [--batch_size 256] ...

线上训练平台会通过环境变量注入数据、checkpoint 和日志路径；本地调试时也可以用
命令行参数覆盖默认值。

常用环境变量：
    TRAIN_DATA_PATH  训练数据目录（*.parquet + schema.json）
    TRAIN_CKPT_PATH  checkpoint 输出目录
    TRAIN_LOG_PATH   日志目录
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import (
    FeatureSchema,
    get_pcvr_data,
    NUM_TIME_BUCKETS,
    ENGINEERED_DENSE_DIM,
    ENGINEERED_DENSE_FEATURE_NAMES,
)
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """把 ``FeatureSchema`` 转成模型侧使用的离散特征规格。

    dataset.py 会把同一组离散特征铺平成一个二维张量。这里记录每个 feature 在
    这个扁平张量里的 ``offset``、``length`` 和词表大小，NS tokenizer 后续按这些
    信息切片、查 Embedding、做多值 pooling。
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_shared_fids(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(fid) for fid in value]
    if isinstance(value, tuple):
        return [int(fid) for fid in value]
    return [int(fid.strip()) for fid in str(value).split(',') if fid.strip()]


def build_shared_fid_tuple_specs(
    user_int_schema: FeatureSchema,
    user_int_vocab_sizes: List[int],
    user_dense_schema: FeatureSchema,
    shared_fids: List[int],
) -> List[Dict[str, int]]:
    user_fid_to_idx = {
        fid: i for i, (fid, _, _) in enumerate(user_int_schema.entries)
    }
    specs: List[Dict[str, int]] = []
    for fid in shared_fids:
        if not user_int_schema.has_feature(fid):
            raise KeyError(f"shared fid {fid} not found in user_int schema")
        if not user_dense_schema.has_feature(fid):
            raise KeyError(f"shared fid {fid} not found in user_dense schema")
        int_offset, int_length = user_int_schema.get_offset_length(fid)
        dense_offset, dense_length = user_dense_schema.get_offset_length(fid)
        if int_length != dense_length:
            raise ValueError(
                f"shared fid {fid} int_length={int_length} != dense_length={dense_length}"
            )
        int_vocab_size = max(user_int_vocab_sizes[int_offset:int_offset + int_length])
        specs.append({
            "fid": fid,
            "int_feature_idx": user_fid_to_idx[fid],
            "int_offset": int_offset,
            "int_length": int_length,
            "int_vocab_size": int_vocab_size,
            "dense_offset": dense_offset,
            "dense_length": dense_length,
        })
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # 路径参数；线上平台通常通过环境变量传入这些路径。
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # 训练循环相关参数。
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # DataLoader 和 Row Group 划分相关参数。
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--valid_num_workers', type=int, default=-1,
                        help='Number of validation DataLoader workers '
                             '(-1 = reuse --num_workers)')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                        help='Write train loss / tqdm postfix and print profiling every N steps')
    parser.add_argument('--max_train_steps', type=int, default=0,
                        help='Profiling-only step cap; 0 means no limit')
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="none",
        choices=["none", "bf16"],
        help="AMP dtype for autocast speed probe. none disables AMP; bf16 enables torch.autocast with bfloat16.",
    )
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # 主模型结构参数。
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # 损失函数参数。
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Embedding 使用独立的稀疏优化器。
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # 控制哪些特征创建 Embedding，以及序列 ID 特征的额外 dropout。
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')
    parser.add_argument('--use_engineered_dense_features', action='store_true', default=True,
                        help='Enable explicit recency + raw sequence length dense features')
    parser.add_argument('--no_engineered_dense_features',
                        dest='use_engineered_dense_features',
                        action='store_false',
                        help='Disable explicit engineered dense features')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS token 的构造方式。
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.add_argument('--use_shared_fid_tuple_token', action='store_true', default=False,
                        help='Enable schema-aware shared user fid int+dense tuple token')
    parser.add_argument('--shared_fids', type=str, default='62,63,64,65,66',
                        help='Comma-separated shared user fids for tuple tokenization')
    parser.add_argument('--shared_fid_tuple_mode', type=str, default='replace',
                        choices=['replace', 'additive'],
                        help='replace excludes shared fids from generic user tokens; '
                             'additive keeps them and adds the tuple token')

    args = parser.parse_args()

    # 线上平台注入的环境变量优先于命令行参数。
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()
    args.shared_fids = parse_shared_fids(args.shared_fids)

    # checkpoint、日志和 TensorBoard 输出目录。
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # 固定随机种子并初始化日志。
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")
    logging.info("Experiment: exp_004_recency_length_dense_features")
    logging.info(f"use_engineered_dense_features={args.use_engineered_dense_features}")
    logging.info(f"engineered_dense_dim={ENGINEERED_DENSE_DIM}")
    logging.info(f"engineered feature names={ENGINEERED_DENSE_FEATURE_NAMES}")
    logging.info(f"amp_dtype={args.amp_dtype}")
    logging.info(f"use_shared_fid_tuple_token={args.use_shared_fid_tuple_token}")
    logging.info(f"shared_fids={args.shared_fids}")
    logging.info(f"shared_fid_tuple_mode={args.shared_fid_tuple_mode}")
    logging.info(
        "original_shared_fids_still_in_generic_tokens="
        f"{args.use_shared_fid_tuple_token and args.shared_fid_tuple_mode == 'additive'}"
    )

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- 数据加载：parquet/schema -> batch tensor ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # 解析 seq_a/seq_b/seq_c/seq_d 的最大截断长度。
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        valid_num_workers=args.valid_num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    # ---- NS 分组：决定离散 user/item 特征进入 token 的顺序和组合 ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- 构建模型：把 schema、tokenizer 配置和 backbone 参数传入 model.py ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)
    shared_fid_tuple_specs: List[Dict[str, int]] = []
    if args.use_shared_fid_tuple_token:
        shared_fid_tuple_specs = build_shared_fid_tuple_specs(
            pcvr_dataset.user_int_schema,
            pcvr_dataset.user_int_vocab_sizes,
            pcvr_dataset.user_dense_schema,
            args.shared_fids,
        )
        for spec in shared_fid_tuple_specs:
            logging.info(
                "shared_fid_tuple spec: "
                f"fid={spec['fid']} "
                f"int_offset={spec['int_offset']} int_length={spec['int_length']} "
                f"dense_offset={spec['dense_offset']} dense_length={spec['dense_length']}"
            )
    logging.info(f"tuple token count={1 if args.use_shared_fid_tuple_token else 0}")

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "use_engineered_dense_features": args.use_engineered_dense_features,
        "engineered_dense_dim": ENGINEERED_DENSE_DIM,
        "use_shared_fid_tuple_token": args.use_shared_fid_tuple_token,
        "shared_fids": args.shared_fids,
        "shared_fid_tuple_mode": args.shared_fid_tuple_mode,
        "shared_fid_tuple_specs": shared_fid_tuple_specs,
    }

    model = PCVRHyFormer(**model_args).to(args.device)
    if args.use_engineered_dense_features:
        logging.info(
            "engineered_alpha initial value="
            f"{float(model.engineered_alpha.detach().cpu().item()):.1f}"
        )

    # 记录 token 数、参数量和 RankMixer 的 T 值，便于排查维度约束。
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"final num_ns={num_ns}, total token count T={T}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- 训练：交给 trainer.py 处理 train/eval/checkpoint ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        max_train_steps=args.max_train_steps,
        amp_dtype=args.amp_dtype,
        train_config=vars(args),
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()

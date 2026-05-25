"""PCVRHyFormer inference script (uploaded by the contestant into the
evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``ns_groups.json`` + ``train_config.json``. All model
hyperparameters are resolved first from the ckpt directory's
``train_config.json`` (written by ``trainer.py`` when saving a checkpoint),
falling back to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent
with the CLI defaults in ``train.py``).

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``global_step``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
#
# These MUST match the argparse defaults in ``train.py``; otherwise once the
# fallback path is actually taken the built model will shape-mismatch the
# saved state_dict.
#
# Special note on ``num_time_buckets``: this value is strictly determined by
# ``dataset.BUCKET_BOUNDARIES`` and is NOT an independent hyperparameter.
# When the feature is enabled we therefore use the constant exposed by the
# dataset module; ``0`` means disabled.
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'num_queries': 1,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.01,
    'seq_top_k': 50,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'rank_mixer_mode': 'full',
    'rank_mixer_moe_num_experts': 8,
    'rank_mixer_moe_top_k': 2,
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 0,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 0,
    'item_ns_tokens': 0,
    'multi_scale_queries': False,
    'q_dropout_mult': 1.0,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (which triggers fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is uniquely determined
    by the length of ``dataset.BUCKET_BOUNDARIES``. Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> derive as ``NUM_TIME_BUCKETS`` or ``0``;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}")
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    """Construct a ``PCVRHyFormer`` from the dataset schema, an NS-groups JSON,
    and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        ns_groups_json: path to the NS-groups JSON file, or ``None`` / empty
            string to disable it (each feature becomes its own singleton group).
        device: torch device.
    """
    # NS grouping. The JSON schema uses *fid* (feature id) values; convert
    # them to positional indices into ``user_int_schema.entries`` /
    # ``item_int_schema.entries`` so ``GroupNSTokenizer`` /
    # ``RankMixerNSTokenizer`` can index ``feature_specs`` directly. This is
    # the same conversion ``train.py`` performs when loading the JSON; doing
    # it here keeps infer.py symmetric with training.
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    # Feature specs.
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.

    Automatically strips ``_orig_mod.`` prefix from checkpoint keys, so that
    checkpoints saved by a torch.compile()-wrapped training script can be
    loaded into an uncompiled inference model.
    """
    state_dict = torch.load(ckpt_path, map_location=device)

    # Strip '_orig_mod.' prefix: torch.compile() wraps the model in an
    # _orig_mod container whose state_dict keys all have this prefix.
    # Training saves the uncompiled state_dict, but as a safety net we
    # handle the prefix here transparently.
    orig_mod_prefix = '_orig_mod.'
    if any(k.startswith(orig_mod_prefix) for k in state_dict.keys()):
        logging.info(
            f"Stripping '{orig_mod_prefix}' prefix from {sum(1 for k in state_dict if k.startswith(orig_mod_prefix))} "
            f"checkpoint keys (torch.compile() _orig_mod container detected)"
        )
        state_dict = {
            (k[len(orig_mod_prefix):] if k.startswith(orig_mod_prefix) else k): v
            for k, v in state_dict.items()
        }

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        expected = set(model.state_dict().keys())
        found = set(state_dict.keys())
        missing = expected - found
        unexpected = found - expected

        diag_parts = [
            "Failed to load state_dict in strict mode.",
            "This usually means the model constructed by build_model does NOT "
            "match the checkpoint.",
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.",
        ]

        if missing:
            # Group by module prefix for readability
            from collections import defaultdict
            by_prefix = defaultdict(list)
            for k in sorted(missing):
                prefix = k.split('.')[0]
                by_prefix[prefix].append(k)
            diag_parts.append(f"--- Missing keys ({len(missing)}) ---")
            for prefix, keys in sorted(by_prefix.items()):
                if len(keys) <= 3:
                    diag_parts.extend(f"  {prefix}: {keys}")
                else:
                    diag_parts.append(f"  {prefix}: {len(keys)} keys (e.g. {keys[0]}, {keys[1]}, ...)")

        if unexpected:
            diag_parts.append(f"--- Unexpected keys ({len(unexpected)}) ---")
            for i, k in enumerate(sorted(unexpected)):
                if i < 5:
                    diag_parts.append(f"  {k}")
            if len(unexpected) > 5:
                diag_parts.append(f"  ... and {len(unexpected) - 5} more")

        for line in diag_parts:
            logging.error(line)
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside the directory pointed at by
    ``$MODEL_OUTPUT_PATH``. Returns ``None`` if no checkpoint is found.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``, handling dynamic seq domains."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        hour=device_batch.get('hour', torch.zeros(device_batch['user_int_feats'].shape[0], dtype=torch.long, device=device)),
        day_of_week=device_batch.get('day_of_week', torch.zeros(device_batch['user_int_feats'].shape[0], dtype=torch.long, device=device)),
        day_of_month=device_batch.get('day_of_month', torch.zeros(device_batch['user_int_feats'].shape[0], dtype=torch.long, device=device)),
    )


def _normalized_histogram(
    values: np.ndarray,
    start: int,
    end: int,
) -> Dict[str, float]:
    """Return normalized mass over a small integer bucket range."""
    if values.size == 0:
        return {}
    total = float(values.size)
    hist: Dict[str, float] = {}
    for v in range(start, end + 1):
        count = int(np.sum(values == v))
        if count > 0:
            hist[str(v)] = count / total
    return hist


def _top_hist_text(hist: Dict[str, float], top_k: int = 4) -> str:
    """Format top normalized buckets for readable logs."""
    if not hist:
        return "n/a"
    items = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return " ".join(f"{k}:{v * 100:.1f}%" for k, v in items)


def _summarize_infer_monitor(
    probs_np: np.ndarray,
    all_hour_list: List[np.ndarray],
    all_dow_list: List[np.ndarray],
    all_dom_list: List[np.ndarray],
    seq_domains: List[str],
    all_seq_lens: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """Build a compact test-side monitor for shift inspection."""
    hours_np = np.concatenate(all_hour_list) if all_hour_list else np.array([], dtype=np.int64)
    dow_np = np.concatenate(all_dow_list) if all_dow_list else np.array([], dtype=np.int64)
    dom_np = np.concatenate(all_dom_list) if all_dom_list else np.array([], dtype=np.int64)

    monitor: Dict[str, Any] = {
        'n_samples': int(len(probs_np)),
        'score': {
            'mean': float(np.mean(probs_np)),
            'std': float(np.std(probs_np)),
            'q05': float(np.percentile(probs_np, 5)),
            'q25': float(np.percentile(probs_np, 25)),
            'q50': float(np.percentile(probs_np, 50)),
            'q75': float(np.percentile(probs_np, 75)),
            'q95': float(np.percentile(probs_np, 95)),
        },
        'calendar': {
            'hour_share': _normalized_histogram(hours_np, 1, 24),
            'weekday_share': _normalized_histogram(dow_np, 1, 7),
            'monthday_share': _normalized_histogram(dom_np, 1, 31),
        },
        'score_by_hour': {},
        'seq_profile': {},
    }
    score_by_hour: Dict[str, Dict[str, float]] = {}
    for hour in range(1, 25):
        mask = hours_np == hour
        if int(mask.sum()) == 0:
            continue
        score_by_hour[str(hour)] = {
            'mean': float(np.mean(probs_np[mask])),
            'std': float(np.std(probs_np[mask])),
            'share': float(np.mean(mask)),
        }
    monitor['score_by_hour'] = score_by_hour
    seq_profile: Dict[str, Dict[str, float]] = {}
    for domain in seq_domains:
        if domain not in all_seq_lens or not all_seq_lens[domain]:
            continue
        lens_np = np.concatenate(all_seq_lens[domain]).astype(np.int64)
        seq_profile[domain] = {
            'mean_len': float(np.mean(lens_np)),
            'p50_len': float(np.percentile(lens_np, 50)),
            'p90_len': float(np.percentile(lens_np, 90)),
            'zero_ratio': float(np.mean(lens_np == 0)),
        }
    monitor['seq_profile'] = seq_profile
    return monitor


def _log_infer_monitor(monitor: Dict[str, Any]) -> None:
    """Print a compact summary of inference-side data and score shape."""
    score = monitor.get('score', {})
    calendar = monitor.get('calendar', {})
    seq_profile = monitor.get('seq_profile', {})

    logging.info(
        "Monitor(test) score mean=%.6f std=%.6f q50=%.6f q95=%.6f n=%d",
        float(score.get('mean', 0.0)),
        float(score.get('std', 0.0)),
        float(score.get('q50', 0.0)),
        float(score.get('q95', 0.0)),
        int(monitor.get('n_samples', 0)),
    )
    logging.info(
        "Monitor(test) calendar hour_top=%s | weekday_top=%s | monthday_top=%s",
        _top_hist_text(calendar.get('hour_share', {})),
        _top_hist_text(calendar.get('weekday_share', {}), top_k=3),
        _top_hist_text(calendar.get('monthday_share', {}), top_k=3),
    )
    if seq_profile:
        seq_parts = []
        for domain in sorted(seq_profile.keys()):
            stats = seq_profile[domain]
            seq_parts.append(
                f"{domain}[mean={stats['mean_len']:.1f},p90={stats['p90_len']:.1f},zero={stats['zero_ratio'] * 100:.1f}%]"
            )
        logging.info("Monitor(test) seq %s", " | ".join(seq_parts))


def _tv_distance(lhs: Dict[str, float], rhs: Dict[str, float]) -> float:
    """Total variation distance between two sparse histograms."""
    keys = set(lhs.keys()) | set(rhs.keys())
    if not keys:
        return 0.0
    return 0.5 * sum(abs(lhs.get(k, 0.0) - rhs.get(k, 0.0)) for k in keys)


def _log_shift_watch(
    model_dir: str,
    test_monitor: Dict[str, Any],
) -> None:
    """Compare current test-side monitor against saved validation monitor."""
    candidates = [
        os.path.join(model_dir, 'eval_monitor.json'),
        os.path.join(os.path.dirname(model_dir.rstrip('/')), 'eval_monitor.json'),
    ]
    valid_monitor = None
    matched_path = None
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, 'r') as f:
                valid_monitor = json.load(f)
            matched_path = candidate
            break

    if valid_monitor is None:
        logging.info("ShiftWatch skipped: eval_monitor.json not found in %s", candidates)
        return

    v_score = valid_monitor.get('score', {})
    t_score = test_monitor.get('score', {})
    logging.info(
        "ShiftWatch source=%s mean_gap=%+.6f std_ratio=%.4f q50_gap=%+.6f q95_gap=%+.6f",
        matched_path,
        float(t_score.get('mean', 0.0)) - float(v_score.get('mean', 0.0)),
        float(t_score.get('std', 0.0)) / max(float(v_score.get('std', 0.0)), 1e-8),
        float(t_score.get('q50', 0.0)) - float(v_score.get('q50', 0.0)),
        float(t_score.get('q95', 0.0)) - float(v_score.get('q95', 0.0)),
    )
    logging.info(
        "ShiftWatch quantile_gap q05=%+.6f q25=%+.6f q75=%+.6f q95=%+.6f iqr_ratio=%.4f",
        float(t_score.get('q05', 0.0)) - float(v_score.get('q05', 0.0)),
        float(t_score.get('q25', 0.0)) - float(v_score.get('q25', 0.0)),
        float(t_score.get('q75', 0.0)) - float(v_score.get('q75', 0.0)),
        float(t_score.get('q95', 0.0)) - float(v_score.get('q95', 0.0)),
        (
            (float(t_score.get('q75', 0.0)) - float(t_score.get('q25', 0.0)))
            / max(float(v_score.get('q75', 0.0)) - float(v_score.get('q25', 0.0)), 1e-8)
        ),
    )

    v_calendar = valid_monitor.get('calendar', {})
    t_calendar = test_monitor.get('calendar', {})
    logging.info(
        "ShiftWatch calendar hour_tv=%.4f weekday_tv=%.4f monthday_tv=%.4f",
        _tv_distance(v_calendar.get('hour_share', {}), t_calendar.get('hour_share', {})),
        _tv_distance(v_calendar.get('weekday_share', {}), t_calendar.get('weekday_share', {})),
        _tv_distance(v_calendar.get('monthday_share', {}), t_calendar.get('monthday_share', {})),
    )
    logging.info(
        "ShiftWatch calendar_top valid_hour=%s | test_hour=%s | valid_weekday=%s | test_weekday=%s",
        _top_hist_text(v_calendar.get('hour_share', {})),
        _top_hist_text(t_calendar.get('hour_share', {})),
        _top_hist_text(v_calendar.get('weekday_share', {}), top_k=3),
        _top_hist_text(t_calendar.get('weekday_share', {}), top_k=3),
    )

    v_seq = valid_monitor.get('seq_profile', {})
    t_seq = test_monitor.get('seq_profile', {})
    seq_parts = []
    for domain in sorted(set(v_seq.keys()) & set(t_seq.keys())):
        v_stats = v_seq[domain]
        t_stats = t_seq[domain]
        seq_parts.append(
            f"{domain}[mean_gap={t_stats['mean_len'] - v_stats['mean_len']:+.2f},"
            f"p90_gap={t_stats['p90_len'] - v_stats['p90_len']:+.2f},"
            f"zero_gap={(t_stats['zero_ratio'] - v_stats['zero_ratio']) * 100:+.2f}%]"
        )
    if seq_parts:
        logging.info("ShiftWatch seq %s", " | ".join(seq_parts))

    v_hour_score = valid_monitor.get('score_by_hour', {})
    t_hour_score = test_monitor.get('score_by_hour', {})
    hour_gaps = []
    for hour in sorted(set(v_hour_score.keys()) & set(t_hour_score.keys()), key=int):
        mean_gap = float(t_hour_score[hour]['mean']) - float(v_hour_score[hour]['mean'])
        std_ratio = float(t_hour_score[hour]['std']) / max(float(v_hour_score[hour]['std']), 1e-8)
        hour_gaps.append((abs(mean_gap), hour, mean_gap, std_ratio, float(t_hour_score[hour]['share'])))
    if hour_gaps:
        hour_gaps.sort(reverse=True)
        parts = []
        for _, hour, mean_gap, std_ratio, share in hour_gaps[:6]:
            parts.append(
                f"h{hour}[gap={mean_gap:+.4f},stdx={std_ratio:.3f},share={share * 100:.1f}%]"
            )
        logging.info("ShiftWatch hour_score %s", " | ".join(parts))


def main() -> None:
    # ---- Read environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Schema: prefer the one from model_dir (to exactly match training);
    #      fall back to the one in data_dir if missing. ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: reuse batch_size / num_workers from training config ----
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)

    # ns_groups_json also comes from training config (e.g. run.sh may have
    # passed an empty string to disable it). When trainer.py has copied the
    # JSON into the ckpt dir, train_config records just the basename, so try
    # resolving against ``model_dir`` first before honoring the raw (possibly
    # absolute) path as a fallback.
    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs = []
    all_user_ids = []
    all_hour_list: List[np.ndarray] = []
    all_dow_list: List[np.ndarray] = []
    all_dom_list: List[np.ndarray] = []
    seq_domains: List[str] = []
    all_seq_lens: Dict[str, List[np.ndarray]] = {}
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if 'hour' in batch:
                all_hour_list.append(batch['hour'].detach().cpu().numpy())
            if 'day_of_week' in batch:
                all_dow_list.append(batch['day_of_week'].detach().cpu().numpy())
            if 'day_of_month' in batch:
                all_dom_list.append(batch['day_of_month'].detach().cpu().numpy())
            if not seq_domains:
                seq_domains = list(batch['_seq_domains'])
                for domain in seq_domains:
                    all_seq_lens[domain] = []
            for domain in seq_domains:
                all_seq_lens[domain].append(
                    batch[f'{domain}_len'].detach().cpu().numpy()
                )

            if (batch_idx + 1) % 100 == 0:
                logging.info(
                    f"Inference progress: batch={batch_idx + 1}, "
                    f"predictions={len(all_probs)}"
                )

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    probs_np = np.asarray(all_probs, dtype=np.float64)
    if probs_np.size > 0:
        test_monitor = _summarize_infer_monitor(
            probs_np=probs_np,
            all_hour_list=all_hour_list,
            all_dow_list=all_dow_list,
            all_dom_list=all_dom_list,
            seq_domains=seq_domains,
            all_seq_lens=all_seq_lens,
        )
        _log_infer_monitor(test_monitor)
        _log_shift_watch(model_dir, test_monitor)

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()

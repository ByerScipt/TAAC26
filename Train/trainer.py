"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import json
import shutil
import contextlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        keep_top_k_best_models: int,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        log_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        grad_accumulation_steps: int = 1,
        use_amp: bool = True,
        amp_dtype: str = 'bf16',
        # 参数滑动平均，只作用在 dense 参数上。
        use_ema: bool = False,
        ema_decay: float = 0.999,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.keep_top_k_best_models: int = max(1, keep_top_k_best_models)
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.log_every_n_steps: int = log_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config

        # Gradient accumulation
        self.grad_accumulation_steps: int = max(1, grad_accumulation_steps)
        self._accum_step: int = 0

        # dense 参数保留一份平滑副本，验证和保存时临时切过去。
        self.use_ema: bool = use_ema
        self.ema_decay: float = ema_decay
        self._ema_shadow: Optional[Dict[str, torch.Tensor]] = None
        self._latest_eval_monitor: Optional[Dict[str, Any]] = None
        self._top_k_best_records: List[Dict[str, Any]] = []

        # AMP (Automatic Mixed Precision)
        self.use_amp: bool = use_amp
        self._amp_dtype_str: str = amp_dtype
        self._amp_dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
        )

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"grad_accumulation_steps={self.grad_accumulation_steps}, "
                     f"keep_top_k_best_models={self.keep_top_k_best_models}, "
                     f"log_every_n_steps={log_every_n_steps}, "
                     f"use_amp={self.use_amp}, amp_dtype={self._amp_dtype_str}, "
                     f"use_ema={self.use_ema}, ema_decay={self.ema_decay}")

    def _embedding_param_ptrs(self) -> "set[int]":
        """收集稀疏表参数地址，用于把 embedding 排除在 EMA 之外。"""
        if not hasattr(self.model, 'get_sparse_params'):
            return set()
        _model = (self.model._orig_mod
                  if hasattr(self.model, '_orig_mod')
                  else self.model)
        return {p.data_ptr() for p in _model.get_sparse_params()}

    def _refresh_weight_average(self) -> None:
        """在优化器完成一次有效更新后，同步 dense 参数的平滑副本。"""
        if not self.use_ema:
            return
        _model = (self.model._orig_mod
                  if hasattr(self.model, '_orig_mod')
                  else self.model)
        sparse_ptrs = self._embedding_param_ptrs()

        if self._ema_shadow is None:
            self._ema_shadow = {}
            for name, param in _model.named_parameters():
                if param.requires_grad and param.data_ptr() not in sparse_ptrs:
                    self._ema_shadow[name] = param.data.clone().detach()
        else:
            with torch.no_grad():
                for name, param in _model.named_parameters():
                    if name in self._ema_shadow:
                        self._ema_shadow[name].mul_(self.ema_decay).add_(
                            param.data, alpha=1.0 - self.ema_decay)

    @contextlib.contextmanager
    def _swap_to_average_weights(self):
        """验证和保存时临时切到平滑权重，退出后恢复训练权重。"""
        if self._ema_shadow is None:
            yield
            return

        _model = (self.model._orig_mod
                  if hasattr(self.model, '_orig_mod')
                  else self.model)

        backup: Dict[str, torch.Tensor] = {}
        for name, param in _model.named_parameters():
            if name in self._ema_shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self._ema_shadow[name])
        try:
            yield
        finally:
            for name, param in _model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _write_eval_monitor(self, ckpt_dir: str) -> None:
        """Persist the latest validation-side monitor next to the checkpoint."""
        if self._latest_eval_monitor is None:
            return
        monitor_path = os.path.join(ckpt_dir, 'eval_monitor.json')
        with open(monitor_path, 'w') as f:
            json.dump(self._latest_eval_monitor, f, indent=2)
        score = self._latest_eval_monitor.get('score', {})
        logging.info(
            "Saved eval monitor to %s (mean=%.6f, std=%.6f)",
            monitor_path,
            float(score.get('mean', 0.0)),
            float(score.get('std', 0.0)),
        )

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            # When torch.compile() is active, self.model.state_dict() returns
            # keys prefixed with '_orig_mod.'; save the uncompiled state_dict
            # so that infer.py (which does NOT use torch.compile) can load it.
            state_dict = (
                self.model._orig_mod.state_dict()
                if hasattr(self.model, '_orig_mod')
                else self.model.state_dict()
            )
            torch.save(state_dict, os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        self._write_eval_monitor(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _candidate_best_dir(self, global_step: int) -> str:
        """Return the canonical on-disk directory for a top-k candidate."""
        return os.path.join(
            self.save_dir,
            self._build_step_dir_name(global_step, is_best=True),
        )

    @staticmethod
    def _normalized_histogram(
        values: np.ndarray,
        start: int,
        end: int,
    ) -> Dict[str, float]:
        """Return a compact normalized histogram over a small integer range."""
        if values.size == 0:
            return {}
        total = float(values.size)
        hist: Dict[str, float] = {}
        for v in range(start, end + 1):
            count = int(np.sum(values == v))
            if count > 0:
                hist[str(v)] = count / total
        return hist

    @staticmethod
    def _top_hist_text(hist: Dict[str, float], top_k: int = 4) -> str:
        """Format the heaviest buckets of a normalized histogram."""
        if not hist:
            return "n/a"
        items = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return " ".join(f"{k}:{v * 100:.1f}%" for k, v in items)

    @staticmethod
    def _hist_peak_share(hist: Dict[str, float]) -> float:
        """Return the dominant bucket share of a normalized histogram."""
        if not hist:
            return 0.0
        return float(max(hist.values()))

    def _build_eval_monitor(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        hours_np: np.ndarray,
        dow_np: np.ndarray,
        dom_np: np.ndarray,
        seq_lens_cat: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Dict[str, Any]:
        """Build a compact validation summary for later shift comparison."""
        monitor: Dict[str, Any] = {
            'epoch': int(epoch),
            'n_samples': int(len(probs)),
            'score': {
                'mean': float(np.mean(probs)),
                'std': float(np.std(probs)),
                'q05': float(np.percentile(probs, 5)),
                'q25': float(np.percentile(probs, 25)),
                'q50': float(np.percentile(probs, 50)),
                'q75': float(np.percentile(probs, 75)),
                'q95': float(np.percentile(probs, 95)),
            },
            'label': {
                'positive_rate': float(np.mean(labels)),
            },
            'calendar': {
                'hour_share': self._normalized_histogram(hours_np, 1, 24),
                'weekday_share': self._normalized_histogram(dow_np, 1, 7),
                'monthday_share': self._normalized_histogram(dom_np, 1, 31),
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
                'mean': float(np.mean(probs[mask])),
                'std': float(np.std(probs[mask])),
                'share': float(np.mean(mask)),
            }
        monitor['score_by_hour'] = score_by_hour
        seq_profile: Dict[str, Dict[str, float]] = {}
        for domain, lens in seq_lens_cat.items():
            lens_np = lens.cpu().numpy().astype(np.int64)
            seq_profile[domain] = {
                'mean_len': float(np.mean(lens_np)),
                'p50_len': float(np.percentile(lens_np, 50)),
                'p90_len': float(np.percentile(lens_np, 90)),
                'zero_ratio': float(np.mean(lens_np == 0)),
            }
        monitor['seq_profile'] = seq_profile
        return monitor

    def _log_eval_monitor(self, monitor: Dict[str, Any]) -> None:
        """Emit a concise validation-side monitor summary."""
        score = monitor.get('score', {})
        label = monitor.get('label', {})
        calendar = monitor.get('calendar', {})
        seq_profile = monitor.get('seq_profile', {})

        logging.info(
            "Monitor(valid) score mean=%.6f std=%.6f q50=%.6f q95=%.6f pos=%.6f",
            float(score.get('mean', 0.0)),
            float(score.get('std', 0.0)),
            float(score.get('q50', 0.0)),
            float(score.get('q95', 0.0)),
            float(label.get('positive_rate', 0.0)),
        )
        logging.info(
            "Monitor(valid) calendar hour_top=%s | weekday_top=%s | monthday_top=%s",
            self._top_hist_text(calendar.get('hour_share', {})),
            self._top_hist_text(calendar.get('weekday_share', {}), top_k=3),
            self._top_hist_text(calendar.get('monthday_share', {}), top_k=3),
        )
        if seq_profile:
            seq_parts = []
            for domain in sorted(seq_profile.keys()):
                stats = seq_profile[domain]
                seq_parts.append(
                    f"{domain}[mean={stats['mean_len']:.1f},p90={stats['p90_len']:.1f},zero={stats['zero_ratio'] * 100:.1f}%]"
                )
            logging.info("Monitor(valid) seq %s", " | ".join(seq_parts))

    def _write_eval_monitor_scalars(self, monitor: Dict[str, Any], global_step: int) -> None:
        """Push shift-anchor scalars into the writer for panel inspection."""
        if self.writer is None:
            return
        score = monitor.get('score', {})
        calendar = monitor.get('calendar', {})
        seq_profile = monitor.get('seq_profile', {})
        self.writer.add_scalar('ShiftAnchor/score_mean', float(score.get('mean', 0.0)), global_step)
        self.writer.add_scalar('ShiftAnchor/score_std', float(score.get('std', 0.0)), global_step)
        self.writer.add_scalar(
            'ShiftAnchor/score_iqr',
            float(score.get('q75', 0.0)) - float(score.get('q25', 0.0)),
            global_step,
        )
        self.writer.add_scalar(
            'ShiftAnchor/hour_peak_share',
            self._hist_peak_share(calendar.get('hour_share', {})),
            global_step,
        )
        self.writer.add_scalar(
            'ShiftAnchor/weekday_peak_share',
            self._hist_peak_share(calendar.get('weekday_share', {})),
            global_step,
        )
        for domain, stats in sorted(seq_profile.items()):
            self.writer.add_scalar(
                f'ShiftAnchor/{domain}_mean_len',
                float(stats.get('mean_len', 0.0)),
                global_step,
            )
            self.writer.add_scalar(
                f'ShiftAnchor/{domain}_zero_ratio',
                float(stats.get('zero_ratio', 0.0)),
                global_step,
            )

    def _score_sort_key(self, record: Dict[str, Any]) -> Tuple[float, int]:
        """Sort top-k records by score descending, then by step descending."""
        return float(record['score']), int(record['global_step'])

    def _should_save_top_k_candidate(self, score: float) -> bool:
        """Return True when ``score`` belongs to the current top-k by AUC."""
        if self.keep_top_k_best_models <= 0:
            return False
        if len(self._top_k_best_records) < self.keep_top_k_best_models:
            return True
        worst = min(self._top_k_best_records, key=self._score_sort_key)
        return score > float(worst['score'])

    def _register_top_k_record(self, score: float, global_step: int, ckpt_dir: str) -> None:
        """Insert or refresh a candidate, then prune on-disk top-k dirs by score."""
        self._top_k_best_records = [
            rec for rec in self._top_k_best_records
            if int(rec['global_step']) != global_step
        ]
        self._top_k_best_records.append({
            'score': float(score),
            'global_step': int(global_step),
            'ckpt_dir': ckpt_dir,
        })
        self._top_k_best_records.sort(key=self._score_sort_key, reverse=True)
        keep = self._top_k_best_records[:self.keep_top_k_best_models]
        remove = self._top_k_best_records[self.keep_top_k_best_models:]
        self._top_k_best_records = keep

        for rec in remove:
            old_dir = rec['ckpt_dir']
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)
                logging.info(
                    "Removed stale best_model dir: %s (score=%.6f)",
                    old_dir,
                    float(rec['score']),
                )

        rank_parts = [
            f"#{idx + 1}@step{rec['global_step']}={rec['score']:.6f}"
            for idx, rec in enumerate(self._top_k_best_records)
        ]
        logging.info("Top-%d best_model scoreboard: %s",
                     self.keep_top_k_best_models, " | ".join(rank_parts))

    def _save_top_k_candidate(
        self,
        global_step: int,
        score: float,
        skip_model_file: bool,
    ) -> None:
        """Persist a checkpoint when its score ranks inside the current top-k."""
        if not self._should_save_top_k_candidate(score):
            return

        with self._swap_to_average_weights():
            ckpt_dir = self._save_step_checkpoint(
                global_step,
                is_best=True,
                skip_model_file=skip_model_file,
            )
        self._register_top_k_record(score, global_step, ckpt_dir)

    # ── Batch / validation helpers ───────────────────────────────────────

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Update early stopping and maintain the top-k validation checkpoints.

        Flow:

        1. Feed the score into ``EarlyStopping`` exactly as before so
           patience / stop behavior keeps tracking only the global best.
        2. Independently check whether the current score belongs to the
           on-disk top-k by validation AUC.
        3. If it does, save a ``*.best_model`` directory and evict the
           current worst member of the top-k set when necessary.
        """
        old_best = self.early_stopping.best_score

        # When torch.compile() is active, pass the uncompiled model to
        # EarlyStopping so that the saved state_dict does NOT include the
        # '_orig_mod.' key prefix.
        _model_for_save = (
            self.model._orig_mod
            if hasattr(self.model, '_orig_mod')
            else self.model
        )

        best_dir = self._candidate_best_dir(total_step)
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        with self._swap_to_average_weights():
            self.early_stopping(val_auc, _model_for_save, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })

        is_new_global_best = (
            old_best is None
            or self.early_stopping.best_score != old_best
        )
        if is_new_global_best and os.path.exists(self.early_stopping.checkpoint_path):
            self._save_top_k_candidate(
                total_step,
                val_auc,
                skip_model_file=True,
            )
        else:
            self._save_top_k_candidate(
                total_step,
                val_auc,
                skip_model_file=False,
            )

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            miniters = self.log_every_n_steps if self.log_every_n_steps > 0 else None
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True, miniters=miniters)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                is_log_step = (
                    self.log_every_n_steps <= 0
                    or total_step % self.log_every_n_steps == 0
                    or step + 1 == len(self.train_loader)
                )
                if self.writer and is_log_step:
                    self.writer.add_scalar('Loss/train', loss, total_step)
                if is_log_step:
                    train_pbar.set_postfix({"loss": f"{loss:.4f}"})
                if self.log_every_n_steps > 0 and is_log_step:
                    logging.info(
                        f"Train step={total_step} epoch={epoch} "
                        f"epoch_step={step + 1}/{len(self.train_loader)} loss={loss:.6f}"
                    )

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
                        if self._latest_eval_monitor is not None:
                            self._write_eval_monitor_scalars(self._latest_eval_monitor, total_step)

                    self._handle_validation_result(total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
                if self._latest_eval_monitor is not None:
                    self._write_eval_monitor_scalars(self._latest_eval_monitor, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            hour=device_batch.get('hour', torch.zeros(device_batch['user_int_feats'].shape[0], dtype=torch.long, device=self.device)),
            day_of_week=device_batch.get('day_of_week', torch.zeros(device_batch['user_int_feats'].shape[0], dtype=torch.long, device=self.device)),
            day_of_month=device_batch.get('day_of_month', torch.zeros(device_batch['user_int_feats'].shape[0], dtype=torch.long, device=self.device)),
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        # Zero gradients only at the start of an accumulation cycle
        if self._accum_step % self.grad_accumulation_steps == 0:
            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()

        # Use AMP autocast when enabled, nullcontext otherwise
        amp_ctx = (
            torch.amp.autocast('cuda', dtype=self._amp_dtype)
            if self.scaler is not None
            else contextlib.nullcontext()
        )
        with amp_ctx:
            model_input = self._make_model_input(device_batch)
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)

        # Normalize loss for gradient accumulation
        loss = loss / self.grad_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Only step optimizer after accumulating enough batches
        if (self._accum_step + 1) % self.grad_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.dense_optimizer)
                if self.sparse_optimizer is not None:
                    self.scaler.unscale_(self.sparse_optimizer)
            # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
            # with certain tensor shapes in this project.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)
            if self.scaler is not None:
                self.scaler.step(self.dense_optimizer)
                if self.sparse_optimizer is not None:
                    self.scaler.step(self.sparse_optimizer)
                self.scaler.update()
            else:
                self.dense_optimizer.step()
                if self.sparse_optimizer is not None:
                    self.sparse_optimizer.step()

            # 每次真实优化器更新后刷新平滑权重。
            self._refresh_weight_average()

        self._accum_step += 1
        return loss.item() * self.grad_accumulation_steps

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []
        all_hour_list = []
        all_dow_list = []
        all_dom_list = []
        seq_domains: list = []
        all_seq_lens: Dict[str, list] = {}

        with torch.no_grad(), self._swap_to_average_weights():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                if 'hour' in batch:
                    all_hour_list.append(batch['hour'].detach().cpu())
                if 'day_of_week' in batch:
                    all_dow_list.append(batch['day_of_week'].detach().cpu())
                if 'day_of_month' in batch:
                    all_dom_list.append(batch['day_of_month'].detach().cpu())
                if step == 0:
                    seq_domains = list(batch['_seq_domains'])
                    for domain in seq_domains:
                        all_seq_lens[domain] = []
                for domain in seq_domains:
                    all_seq_lens[domain].append(batch[f'{domain}_len'].detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        if len(probs) > 0:
            hours_np = (
                torch.cat(all_hour_list, dim=0).numpy()[~nan_mask]
                if all_hour_list else np.array([], dtype=np.int64)
            )
            dow_np = (
                torch.cat(all_dow_list, dim=0).numpy()[~nan_mask]
                if all_dow_list else np.array([], dtype=np.int64)
            )
            dom_np = (
                torch.cat(all_dom_list, dim=0).numpy()[~nan_mask]
                if all_dom_list else np.array([], dtype=np.int64)
            )
            seq_lens_cat = {
                domain: torch.cat(chunks, dim=0)
                for domain, chunks in all_seq_lens.items()
            }
            self._latest_eval_monitor = self._build_eval_monitor(
                probs=probs,
                labels=labels_np,
                hours_np=hours_np,
                dow_np=dow_np,
                dom_np=dom_np,
                seq_lens_cat=seq_lens_cat,
                epoch=epoch,
            )
            self._log_eval_monitor(self._latest_eval_monitor)
        else:
            self._latest_eval_monitor = None

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label

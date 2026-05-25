"""PCVRHyFormer 的训练循环。

这里负责把 DataLoader 产出的 batch 送进模型、计算 loss、反向传播、验证 AUC，
并保存最优 checkpoint。类名里保留了历史上的 "Ranking" 后缀，实际训练目标是
pointwise 二分类。
"""

import os
import glob
import shutil
import logging
import time
from typing import Any, Dict, Optional, Tuple

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
    """封装一次完整的 pointwise 二分类训练。

    batch 中的主要字段如下：
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d，以及每一路对应的 *_len 和 *_time_bucket
    - engineered_dense_feats
    - label

    训练 loss 支持 BCEWithLogitsLoss 和 Focal Loss；验证指标是 AUC 和 logloss。
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
        log_every_n_steps: int = 50,
        max_train_steps: int = 0,
        amp_dtype: str = "none",
        train_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # checkpoint 旁边会复制 schema.json，infer.py 用它还原训练时的特征布局。
        self.schema_path: Optional[str] = schema_path
        # 如果使用 ns_groups.json，也复制到 checkpoint 目录，保证 checkpoint 自包含。
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Embedding 参数和稠密网络参数分开优化：前者用 Adagrad，后者用 AdamW。
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
        self.log_every_n_steps: int = max(1, log_every_n_steps)
        self.max_train_steps: int = max(0, max_train_steps)
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_dtype != "none" and str(self.device).startswith("cuda")
        self.train_config: Optional[Dict[str, Any]] = train_config
        self._logged_engineered_dense_sanity = False

        logging.info(f"amp_dtype={self.amp_dtype}")
        if self.amp_dtype == "bf16" and self.amp_enabled:
            if not torch.cuda.is_bf16_supported():
                logging.warning("BF16 AMP requested but torch.cuda.is_bf16_supported() is False; disabling AMP.")
                self.amp_enabled = False
            else:
                logging.info("BF16 AMP enabled.")
        else:
            logging.info("AMP disabled.")

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}")

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """根据 global step 和模型关键配置生成 checkpoint 子目录名。

        例子：``global_step2500.layer=2.head=4.hidden=64.best_model``。
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
        """把推理所需的配置文件写到 checkpoint 目录。

        ``model.pt`` 只包含参数。为了让 infer.py 能构建同样的数据 schema 和模型
        配置，这里额外保存：

        - ``schema.json``：特征布局。
        - ``ns_groups.json``：NS token 分组配置。
        - ``train_config.json``：训练时的命令行参数和模型开关。

        当 ``ns_groups.json`` 被复制进 checkpoint 目录时，``train_config.json`` 里
        的路径会改写成文件名，方便在不同机器上移动 checkpoint。
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
                # 保存相对文件名，infer.py 会在 checkpoint 目录下解析它。
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """保存一个 step checkpoint。

        参数：
            global_step: 当前训练 step。
            is_best: 是否在目录名后追加 ``.best_model``。
            skip_model_file: EarlyStopping 已经写出 ``model.pt`` 时设为 True，只补写
                schema/config 等配套文件。

        返回：
            checkpoint 目录路径。
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """删除旧 best_model 目录，磁盘上只保留当前最优模型。"""
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """把模型会用到的 tensor 移到训练设备。

        batch 中还有 ``timestamp``、``user_id`` 等元信息，这些字段不参与 forward，
        保持在 CPU 可以减少无意义的数据搬运。
        """
        seq_domains = batch['_seq_domains']
        needed_tensor_keys = {
            'user_int_feats',
            'user_dense_feats',
            'item_int_feats',
            'item_dense_feats',
            'engineered_dense_feats',
            'label',
        }
        for domain in seq_domains:
            needed_tensor_keys.add(domain)
            needed_tensor_keys.add(f'{domain}_len')
            needed_tensor_keys.add(f'{domain}_time_bucket')

        device_batch: Dict[str, Any] = {'_seq_domains': seq_domains}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and k in needed_tensor_keys:
                device_batch[k] = v.to(self.device, non_blocking=True)
            elif not isinstance(v, torch.Tensor) or k not in needed_tensor_keys:
                device_batch[k] = v
        return device_batch

    def _log_engineered_dense_sanity(self, device_batch: Dict[str, Any]) -> None:
        if self._logged_engineered_dense_sanity:
            return
        feats = device_batch.get('engineered_dense_feats')
        if feats is None:
            return
        x = feats.detach().float()
        logging.info(
            "engineered_dense_feats sanity: "
            f"shape={tuple(x.shape)} "
            f"mean={x.mean().item():.6f} "
            f"std={x.std(unbiased=False).item():.6f} "
            f"min={x.min().item():.6f} "
            f"max={x.max().item():.6f} "
            f"nan_count={int(torch.isnan(x).sum().item())} "
            f"inf_count={int(torch.isinf(x).sum().item())}"
        )
        self._logged_engineered_dense_sanity = True

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """处理一次验证结果，并在出现新最佳时保存 checkpoint。

        这里先判断本次 AUC 是否有机会刷新最佳值。只有有机会刷新时，才会清理旧的
        ``best_model`` 目录并让 EarlyStopping 写 ``model.pt``。确认模型文件已经
        写出后，再补写 schema 和 config，避免出现只有配置文件、没有模型参数的目录。

        这种顺序让 checkpoint 目录始终保持可推理状态。
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # 本次分数不会刷新最佳值，磁盘上的现有 best_model 保持不动。
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # 当前 step 有机会刷新最佳值，先把 EarlyStopping 的输出路径切到规范目录。
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # 先清理旧 best 目录，随后由 EarlyStopping 写入新的 model.pt。
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        # 确认 model.pt 存在后再补写配套文件，保证目录可以直接用于推理。
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def _log_train_profile(
        self,
        total_step: int,
        step_time_sum: float,
        batch_to_device_time_sum: float,
        fwd_bwd_opt_time_sum: float,
        samples: int,
        steps: int,
    ) -> None:
        """记录最近一段 step 的吞吐、耗时和显存峰值。"""
        if steps <= 0:
            return
        samples_per_sec = samples / step_time_sum if step_time_sum > 0 else 0.0
        max_gpu_allocated_mb = 0.0
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            max_gpu_allocated_mb = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            )
        logging.info(
            "[TrainProfile] "
            f"step={total_step} interval_steps={steps} "
            f"avg_step_time={step_time_sum / steps:.6f}s "
            f"avg_batch_to_device_time={batch_to_device_time_sum / steps:.6f}s "
            f"avg_forward_backward_optimizer_time={fwd_bwd_opt_time_sum / steps:.6f}s "
            f"samples_per_sec={samples_per_sec:.2f} "
            f"max_gpu_allocated_mb={max_gpu_allocated_mb:.2f}"
        )
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(self.device)

    def train(self) -> None:
        """执行完整训练流程。

        每个 batch 会依次经过：搬到设备、构造 ``ModelInput``、forward、loss、
        backward、梯度裁剪、两个 optimizer 更新。验证可以按 step 触发，也可以在
        每个 epoch 结束后触发。
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0
            profile_steps = 0
            profile_samples = 0
            profile_step_time_sum = 0.0
            profile_batch_to_device_time_sum = 0.0
            profile_fwd_bwd_opt_time_sum = 0.0

            for step, batch in train_pbar:
                (
                    loss,
                    step_time,
                    batch_to_device_time,
                    fwd_bwd_opt_time,
                    sample_count,
                ) = self._train_step(batch)
                total_step += 1
                loss_sum += loss
                profile_steps += 1
                profile_samples += sample_count
                profile_step_time_sum += step_time
                profile_batch_to_device_time_sum += batch_to_device_time
                profile_fwd_bwd_opt_time_sum += fwd_bwd_opt_time

                if self.writer and total_step % self.log_every_n_steps == 0:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                if total_step % self.log_every_n_steps == 0:
                    train_pbar.set_postfix({"loss": f"{loss:.4f}"})
                    self._log_train_profile(
                        total_step=total_step,
                        step_time_sum=profile_step_time_sum,
                        batch_to_device_time_sum=profile_batch_to_device_time_sum,
                        fwd_bwd_opt_time_sum=profile_fwd_bwd_opt_time_sum,
                        samples=profile_samples,
                        steps=profile_steps,
                    )
                    profile_steps = 0
                    profile_samples = 0
                    profile_step_time_sum = 0.0
                    profile_batch_to_device_time_sum = 0.0
                    profile_fwd_bwd_opt_time_sum = 0.0

                if self.max_train_steps > 0 and total_step >= self.max_train_steps:
                    if profile_steps > 0:
                        self._log_train_profile(
                            total_step=total_step,
                            step_time_sum=profile_step_time_sum,
                            batch_to_device_time_sum=profile_batch_to_device_time_sum,
                            fwd_bwd_opt_time_sum=profile_fwd_bwd_opt_time_sum,
                            samples=profile_samples,
                            steps=profile_steps,
                        )
                    logging.info(
                        f"Reached max_train_steps={self.max_train_steps}; "
                        "stopping before validation/checkpoint for speed probe"
                    )
                    return

                # step 级验证，用于较长 epoch 中提前观察 AUC 曲线。
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

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

            self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # 达到配置 epoch 后，周期性重置高基数 Embedding，降低 ID 特征过拟合。
            # 参考：
            # KuaiShou Tech., "MultiEpoch: Reusing Training Data for
            # Click-Through Rate Prediction", https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # 按参数地址保存 Adagrad 状态。高基数 Embedding 会被重置，低基数
                # Embedding 的优化器状态会在新 optimizer 上恢复。
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
                # 重建后只恢复未重置参数的优化器状态。
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """把 batch 字典整理成模型 forward 需要的 ``ModelInput``。"""
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
            engineered_dense_feats=device_batch.get('engineered_dense_feats'),
        )

    def _train_step(self, batch: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
        """执行一个训练 step，并返回 loss、耗时和样本数。"""
        step_t0 = time.perf_counter()
        move_t0 = time.perf_counter()
        device_batch = self._batch_to_device(batch)
        batch_to_device_time = time.perf_counter() - move_t0
        self._log_engineered_dense_sanity(device_batch)
        label = device_batch['label'].float()
        sample_count = int(label.numel())

        fwd_bwd_opt_t0 = time.perf_counter()
        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        amp_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float32
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=self.amp_enabled,
        ):
            model_input = self._make_model_input(device_batch)
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        # foreach=False 是本项目的稳定性设置，避免特定形状下的 CUDA kernel 异常。
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        fwd_bwd_opt_time = time.perf_counter() - fwd_bwd_opt_t0
        step_time = time.perf_counter() - step_t0
        return loss.item(), step_time, batch_to_device_time, fwd_bwd_opt_time, sample_count

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """跑完整个验证集，返回 AUC 和 logloss。

        指标计算前会过滤 NaN 预测，避免一次异常 batch 让整次验证报错。
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().float().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0).float()
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # sklearn 负责计算二分类 AUC。
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # 过滤 NaN 预测。
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

        # logloss 使用和 AUC 一致的 NaN 过滤口径。
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行一个验证 step，返回当前 batch 的 logits 和 labels。"""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        amp_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float32
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=self.amp_enabled,
        ):
            model_input = self._make_model_input(device_batch)
            logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
            logits = logits.squeeze(-1).float()  # (B,)

        return logits, label

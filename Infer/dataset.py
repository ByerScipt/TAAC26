"""PCVR Parquet 数据读取模块，针对训练吞吐做过优化。

模块直接读取官方多列 Parquet 文件，并从 ``schema.json`` 解析特征布局。

主要优化点：
- 预分配 numpy buffer，减少 batch 内反复创建数组的开销。
- 序列特征直接写入三维 buffer，减少 padding 和 stack 的中间对象。
- 提前建立列名到列号的映射，避免每个 batch 反复查字符串。
- DataLoader 多 worker 场景使用 ``file_system`` tensor 共享策略，降低
  ``/dev/shm`` 空间不足带来的风险。
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing 从 numpy 1.20 开始可用。旧版本 numpy 下用一个空 shim 保持类型
# 标注可解析，避免导入阶段因为 ``npt.NDArray[np.int64]`` 报错。
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── 特征 Schema ──────────────────────────────────


class FeatureSchema:
    """记录每个特征在扁平张量中的位置。

    ``entries`` 中每一项都是 ``(feature_id, offset, length)``。下游代码通过
    ``feature_id`` 找到对应切片，再交给 Embedding、dense projection 或 tuple
    tokenizer 使用。

    int 特征的长度规则：
      - int_value: length = 1
      - int_array: length = 数组长度
      - int_array_and_float_array: length = int 部分长度

    dense 特征的长度规则：
      - float_value: length = 1
      - float_array: length = 数组长度
      - int_array_and_float_array: length = float 部分长度
    """

    def __init__(self) -> None:
        # 按 schema 顺序保存 (feature_id, offset, length)。
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # fid 到 (offset, length) 的快速索引。
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """把一个特征追加到 schema 末尾。"""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """按 feature_id 查询 ``(offset, length)``。"""
        return self._fid_to_entry[feature_id]

    def has_feature(self, feature_id: int) -> bool:
        """判断 schema 中是否存在指定 feature_id。"""
        return feature_id in self._fid_to_entry

    @property
    def feature_ids(self) -> List[int]:
        """按插入顺序返回所有 feature_id。"""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """序列化成普通 dict，方便写入 JSON。"""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """从 dict 还原 ``FeatureSchema``。"""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# 多 worker DataLoader 下使用文件系统共享 tensor，缓解 /dev/shm 空间不足。
torch.multiprocessing.set_sharing_strategy('file_system')

# 时间差分桶边界。0 表示 padding，1..64 表示有效时间差桶。
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# time-bucket Embedding 的槽位数，包含 padding=0。
#
# 该常量由 BUCKET_BOUNDARIES 的长度唯一决定。model.py 里的
# ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` 必须与这里一致，否则运行时
# 会出现 Embedding 索引越界。
#
# 因此 train.py / infer.py 只暴露 ``--use_time_buckets`` 开关，具体桶数统一从这里派生。
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1

SEQ_CALENDAR_FEATURE_NAMES = (
    'hour_of_day',
    'day_of_week',
    'day_of_month',
)
SEQ_CALENDAR_FEATURE_DIM = len(SEQ_CALENDAR_FEATURE_NAMES)
LOCAL_TIME_OFFSET_SECONDS = 8 * 3600

ENGINEERED_DENSE_DOMAINS = ('seq_a', 'seq_b', 'seq_c', 'seq_d')
ENGINEERED_DENSE_FEATURE_NAMES = [
    f'{domain}_{feature}'
    for domain in ENGINEERED_DENSE_DOMAINS
    for feature in (
        'log1p_len',
        'log1p_last_gap',
        'missing_history',
        'gap_le_60',
        'gap_le_3600',
        'gap_le_86400',
    )
]
ENGINEERED_DENSE_DIM = len(ENGINEERED_DENSE_FEATURE_NAMES)


class PCVRParquetDataset(IterableDataset):
    """直接读取官方多列 Parquet 的 IterableDataset。

    输出 batch 已经是模型可消费的 tensor 字典：
    - int 特征：标量或 list，多值特征按 padding=0 补齐，所有 <=0 的值映射为 0。
    - dense 特征：``list<float>``，按 schema 中的最大长度补齐。
    - sequence 特征：按 seq_a/seq_b/seq_c/seq_d 分域组织，形状为 ``[B, S, L]``。
    - time bucket：由当前样本 timestamp 与序列事件 timestamp 的差值分桶得到。
    - label：训练时由 ``label_type == 2`` 映射得到二分类标签。
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        return_user_id: bool = False,
    ) -> None:
        """初始化 parquet 文件、schema 和可复用 buffer。

        参数：
            parquet_path: parquet 文件目录，或单个 parquet 文件路径。
            schema_path: 描述特征布局的 schema JSON 路径。
            batch_size: 预分配 buffer 使用的固定 batch size。
            seq_max_lens: 每个序列域的截断长度覆盖配置，例如 ``{'seq_d': 256}``。
                未配置的序列域使用默认长度 256。
            shuffle: 是否在 ``buffer_batches`` 个 batch 的窗口内打乱样本。
            buffer_batches: shuffle buffer 大小，单位是 batch。
            row_group_range: Row Group 的 ``(start, end)`` 切片；为 ``None`` 时使用
                全部 Row Group。
            clip_vocab: 为 True 时越界 ID 会裁剪为 0；为 False 时直接报错。
            is_training: 为 True 时从 ``label_type == 2`` 生成 label；推理时返回
                全 0 label 占位。
            return_user_id: 为 True 时返回原始 user_id。训练阶段保持 False，减少
                Python list 转换开销。
        """
        super().__init__()

        # parquet_path 支持目录和单文件两种形式。
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.return_user_id = return_user_id
        # 越界 ID 统计，格式：
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # 建立所有 Row Group 的索引列表。
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # 读取 schema.json，并建立各类特征的布局。
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- 提前建立列名到列号的映射 ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- 预分配 numpy buffer ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_calendar = {}
        self._buf_seq_lens = {}
        self._buf_engineered_dense = np.zeros((B, ENGINEERED_DENSE_DIM), dtype=np.float32)
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_calendar[domain] = np.zeros(
                (B, SEQ_CALENDAR_FEATURE_DIM, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- 为 int 列提前生成读取计划：(col_idx, offset, vocab_size) ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # 序列列读取计划：{domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """从 ``schema_path`` 解析 user/item/sequence 的特征布局。"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense 当前为空，保留 schema 位置以兼容模型接口 ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains：解析 seq_a/seq_b/seq_c/seq_d 的 sideinfo 和 timestamp ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len 来自命令行覆盖配置，未配置的序列域使用 256。
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # 按 Row Group 逐个向上取整，得到 batch 数上界。
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """合并 shuffle buffer 中的 batch，按样本粒度打乱后再切回 batch。
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- 辅助函数 ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """记录 ID 越界情况，并按配置把越界值裁剪成 0。

        统计先缓存在内存里，避免训练过程中频繁向控制台打印。
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """输出 ID 越界统计。

        传入 ``path`` 时写文件；未传入时写入 ``logging.info``。
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """把 Arrow ``ListArray`` 中的 int list 补齐成 ``[B, max_len]``。

        所有 <=0 的值都会映射为 padding=0。原始数据中的 -1 表示缺失，这里和 0
        使用同一套 padding 语义。

        返回：
            ``(padded, lengths)``。``padded`` 形状为 ``[B, max_len]``，
            ``lengths`` 形状为 ``[B]``。
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # 兼容旧脚本 bench_raw_dataset.py 和重命名前的外部调用。新代码直接调用
    # `_pad_varlen_int_column`。
    _pad_varlen_column = _pad_varlen_int_column

    def _compute_engineered_dense_feats(
        self,
        batch: "pa.RecordBatch",
        timestamps: "npt.NDArray[np.int64]",
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """计算显式 recency 和原始序列长度 dense 特征。"""
        feats = self._buf_engineered_dense[:B]
        feats[:] = 0.0

        for domain_idx, domain in enumerate(ENGINEERED_DENSE_DOMAINS):
            side_plan, ts_ci = self._seq_plan[domain]
            base = domain_idx * 6

            if ts_ci is None:
                fallback_ci = side_plan[0][0]
                fallback_col = batch.column(fallback_ci)
                fallback_offs = fallback_col.offsets.to_numpy()
                for i in range(B):
                    raw_seq_len = int(fallback_offs[i + 1]) - int(fallback_offs[i])
                    feats[i, base] = np.log1p(max(raw_seq_len, 0))
                    feats[i, base + 2] = 1.0
                continue

            ts_col = batch.column(ts_ci)
            ts_offs = ts_col.offsets.to_numpy()
            ts_vals = ts_col.values.to_numpy()

            for i in range(B):
                start = int(ts_offs[i])
                end = int(ts_offs[i + 1])
                raw_seq_len = max(end - start, 0)
                row_ts = ts_vals[start:end]
                valid_ts = row_ts[row_ts > 0]

                feats[i, base] = np.log1p(raw_seq_len)
                if valid_ts.size == 0:
                    feats[i, base + 2] = 1.0
                    continue

                last_ts = int(valid_ts.max())
                last_gap = max(int(timestamps[i]) - last_ts, 0)
                feats[i, base + 1] = np.log1p(last_gap)
                feats[i, base + 3] = 1.0 if last_gap <= 60 else 0.0
                feats[i, base + 4] = 1.0 if last_gap <= 3600 else 0.0
                feats[i, base + 5] = 1.0 if last_gap <= 86400 else 0.0

        return feats

    def _compute_seq_calendar_features(
        self,
        ts_padded: "npt.NDArray[np.int64]",
    ) -> "npt.NDArray[np.int64]":
        """Convert per-event unix timestamps into compact local-calendar ids.

        Output shape is ``[B, 3, L]``. Every feature reserves id=0 for padding:
        hour 1..24, day-of-week 1..7, day-of-month 1..31.
        Timestamps are shifted to UTC+8 before calendar extraction
        to match the competition logs' Beijing-time interpretation.
        """
        B, L = ts_padded.shape
        calendar = np.zeros((B, SEQ_CALENDAR_FEATURE_DIM, L), dtype=np.int64)
        valid = ts_padded > 0
        if not valid.any():
            return calendar

        local_seconds = ts_padded[valid] + LOCAL_TIME_OFFSET_SECONDS
        days = local_seconds // 86400
        seconds_in_day = local_seconds % 86400

        hour = seconds_in_day // 3600
        # 1970-01-01 was Thursday. Monday=0, Sunday=6.
        day_of_week = (days + 3) % 7

        months = local_seconds.astype('datetime64[s]').astype('datetime64[M]')
        day_of_month = (
            local_seconds.astype('datetime64[s]').astype('datetime64[D]')
            - months.astype('datetime64[D]')
        ).astype(np.int64)

        calendar[:, 0, :][valid] = hour + 1
        calendar[:, 1, :][valid] = day_of_week + 1
        calendar[:, 2, :][valid] = day_of_month + 1
        return calendar

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """把 Arrow ``ListArray<float>`` 补齐成 ``[B, max_dim]``。"""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """把 Arrow RecordBatch 转成模型训练可直接使用的 tensor 字典。"""
        B = batch.num_rows

        # ---- meta：样本时间戳和二分类标签 ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)

        # ---- user_int：写入预分配 buffer ----
        # null 会通过 fill_null 变成 0，-1 会通过 arr<=0 变成 0。缺失值和 padding
        # 共享 ID=0。vs==0 表示 schema 缺少词表信息，dataset 侧整列置 0，保证模型
        # 中 1 槽 Embedding 不会越界。
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int：处理方式和 user_int 一致 ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense：变长 float list 补齐后写入扁平 dense 张量 ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'engineered_dense_feats': torch.from_numpy(
                self._compute_engineered_dense_feats(batch, timestamps, B).copy()),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            '_seq_domains': self.seq_domains,
        }
        if self.return_user_id:
            result['user_id'] = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- Sequence features：直接把每个序列域写入三维 buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # 直接复用预分配的三维 buffer，形状为 [B, S, L]。
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # 先收集每个 side-info 列的 offsets、values、词表大小和列号，再统一填充。
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # 所有 <=0 的序列取值映射到 padding=0。
            out[out <= 0] = 0

            # 按每个 side-info 特征的 vocab_size 检查越界。vs==0 表示没有词表信息，
            # 该特征整片置 0，避免模型侧 Embedding 越界。
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # 时间差分桶：当前样本 timestamp 减去序列事件 timestamp。
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            calendar_feats = self._buf_seq_calendar[domain][:B]
            calendar_feats[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # 把事件 timestamp 补齐成 (B, max_len)。
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted 的原始结果范围是 [0, len(BUCKET_BOUNDARIES)]。
                # 加 1 后作为有效 bucket id。超过最大边界的时间差会裁剪到最后一个
                # bucket，保证最终索引落在 time_embedding 的有效范围内。
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets
                calendar_feats[:] = self._compute_seq_calendar_features(ts_padded)

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            result[f'{domain}_calendar_feats'] = torch.from_numpy(calendar_feats.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    valid_num_workers: int = -1,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    return_user_id: bool = False,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """从官方多列 Parquet 文件创建 train / valid DataLoader。

    验证集取文件顺序下最后 ``valid_ratio`` 比例的 Row Group。当前比赛主线保留
    baseline 的原始划分方式，本地 AUC 和线上分数会因此存在偏差，但训练能覆盖更新的
    数据分布。

    返回：
        ``(train_loader, valid_loader, train_dataset)``。第三个返回值提供
        ``user_int_schema``、``item_int_schema`` 等信息，train.py 用它构建模型。
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    train_end = total_rgs - n_valid_rgs
    n_train_use_rgs = train_end

    # train_ratio 用于只取训练 Row Group 的前 N%，主要服务快速 smoke test。
    if train_ratio < 1.0:
        n_train_use_rgs = max(1, int(train_end * train_ratio))
        logging.info(
            f"train_ratio={train_ratio}: using {n_train_use_rgs}/{train_end} "
            "pre-validation train Row Groups")

    train_range = (0, n_train_use_rgs)
    valid_range = (train_end, total_rgs)

    train_rows = sum(r[2] for r in rg_info[train_range[0]:train_range[1]])
    valid_rows = sum(r[2] for r in rg_info[valid_range[0]:valid_range[1]])

    use_cuda = torch.cuda.is_available()
    if valid_num_workers < 0:
        valid_num_workers = num_workers

    logging.info(
        "Row Group split: "
        f"total_rgs={total_rgs}, "
        f"train_range={train_range}, "
        f"valid_range={valid_range}, "
        f"train_rows={train_rows}, "
        f"valid_rows={valid_rows}, "
        f"num_workers={num_workers}, "
        f"valid_num_workers={valid_num_workers}")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=train_range,
        clip_vocab=clip_vocab,
        return_user_id=return_user_id,
    )

    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    _valid_kw = {}
    if valid_num_workers > 0:
        _valid_kw['persistent_workers'] = True
        _valid_kw['prefetch_factor'] = 2

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=valid_range,
        clip_vocab=clip_vocab,
        return_user_id=return_user_id,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=valid_num_workers, pin_memory=use_cuda, **_valid_kw,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}, "
                 f"num_workers={num_workers}, valid_num_workers={valid_num_workers}")

    return train_loader, valid_loader, train_dataset

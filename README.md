# TAAC 2026 比赛方案

[English](README_EN.md)

这是我们队伍在 TAAC 2026 比赛中的开源整理版本。仓库保持干净的主线提交历史：从官方 PyTorch baseline 出发，只包含最终决定保留的实验改进。每个 commit 都记录了 `Why / What changed / Impact`，以及本地和线上 AUC。

失败实验未收录到公开历史与实验表中。其中一些影响了后续设计方向（尤其是 `core_refactor`），但仓库仅保留主线改进。

## 仓库结构

```text
.
├── official_baseline/      # 官方 PyTorch baseline，作为参考
├── Train/                  # 最终主线模型的训练代码
├── Infer/                  # 最终主线模型的推理代码
├── TAAC_experiments.csv    # 公开主线实验记录
├── README.md              # 中文说明
└── README_EN.md           # English README
```

前两个公开提交（`baseline` 和 `bf16_amp`）沿用官方 baseline 的根目录结构，尚未拆分 `Train/` 和 `Infer/`。从 `recency_dense` 开始，公开代码拆分为 `official_baseline/`、`Train/`、`Infer/` 三个目录。

## 主线结果

| Step | Experiment | Key idea | Local AUC | Local loss | Online AUC |
|---:|---|---|---:|---:|---:|
| 0 | baseline | 官方 PyTorch baseline | 0.86219 | 0.22466 | 0.812646 |
| 1 | bf16_amp | BF16 混合精度训练 | 0.86247 | 0.22444 | 0.81223 |
| 2 | recency_dense | 按 domain 加 recency/length dense 特征 | 0.8619777 | 0.224132 | 0.814566 |
| 3 | tuple_tokens | 对字段 62-66 建模 tuple token | 0.864002 | 0.22274 | 0.816017 |
| 4 | calendar_tokens | 序列 calendar token，去掉 month-of-year | 0.8655957 | 0.221923 | 0.821676 |
| 5 | time_din | 用当前时间调制 DIN 序列池化 | 0.8666739 | 0.2227302 | 0.825175 |
| 6 | multi_scale_queries | 短期、中期、长期多尺度 query | 0.867078 | 0.221575 | 0.827658 |
| 7 | time_tokens | 显式当前时间 NS token | 0.8671137 | 0.220959 | 0.828348 |
| 8 | core_refactor | 基于额外消融重构核心模型 | 0.8675917 | 0.253418 | 0.829255 |
| 9 | sparse_moe | token-level sparse MoE | 0.86811 | 0.254258 | 0.829527 |
| 10 | feature_tokens | 按特征类型拆分 user-dense tokenization | 0.869321 | 0.253492 | 0.829901 |
| 11 | final_no_moe | 保留异质 tokenization，默认关闭 MoE | 0.868445 | 0.255211 | 0.830774 |

## 实验说明

### baseline

官方 PyTorch baseline 是所有实验的起点，保留原始 schema、metric、submission 格式和推理协议。

### bf16_amp

加入 BF16 AMP 来提升迭代速度。单轮训练时间从约 67 分钟降到 42 分钟，线上 AUC 有小幅损失。

### recency_dense

为每个 domain 加入低成本 dense 统计特征：

- `log1p(seq_len)`
- `log1p(last_gap)`
- missing flag
- `gap <= 60`, `gap <= 3600`, `gap <= 86400`

总共 24 维，通过 zero-init gate 融合。

### tuple_tokens

字段 `62/63/64/65/66` 同时包含 integer 和 dense 信号，这里选择建模为共享 tuple token。

### calendar_tokens

将序列 timestamp 展开为 calendar token。保留 hour-of-day、day-of-week、day-of-month；month-of-year 在消融后没有保留。

### time_din

在 DIN 式序列池化中引入当前时间上下文，让模型在候选 item 和当前时间的双重条件下，更精准地选择相关历史行为。

### multi_scale_queries

用 short/mid/long 三种 query view 解码用户兴趣，兼顾近期行为与长期偏好。

### time_tokens

将当前时间作为显式的 non-sequence token 注入模型，使模型可以直接 attend 到时间上下文。

### core_refactor

这是一次较大的主线重构，基于未公开的模块新增/消融实验。为保持主线清晰，失败实验未展开描述。

这一步从 baseline 出发，重新整合已验证有效的模块：

- short/mid/long multi-scale query decomposition
- domain-specific current-time KV conditioning
- focal loss
- user dense BatchNorm
- dense-only EMA
- stronger query dropout

这是公开主线里第一次把线上 AUC 推到 `0.829+` 的 checkpoint。

### sparse_moe

加入极简的 token-level sparse MoE：每个 token 在 8 个专家中选择 top-2，用于应对 domain 和 token 层面的异质性。

### feature_tokens

按特征类型拆分 user-dense tokenization：

- 字段 `61` 和 `87`：embedding-like dense token
- 字段 `62-66`：stat-bucket dense features，先 `clamp` 再 `log1p` 后投影
- 字段 `89-91`：quantile-style feature token

### final_no_moe

最终消融实验表明，保留异质 feature tokenization 并默认关闭 sparse MoE，线上效果更好。MoE 仍作为可选开关保留，方便复现与受控消融。

## 训练

```bash
cd Train
bash run.sh
```

数据路径依赖 TAAC 官方运行环境。若本地路径不同，需要通过 `run.sh` 追加参数或修改数据路径参数。

## 推理

```bash
cd Infer
python infer.py
```

推理脚本使用官方评测容器约定的环境变量：

```text
MODEL_OUTPUT_PATH   checkpoint 目录
EVAL_DATA_PATH      测试数据目录
EVAL_RESULT_PATH    predictions.json 输出目录
```

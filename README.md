<a id="chinese"></a>

# TAAC 2026 比赛方案

[English](#english)


最终分数0.83077，未能进复赛，不过已经是很满意的分数了。我在main分支按照我们实际上分的顺序进行了多次提交，以体现真实的更新过程。失败的实验就不开源了，不然太乱了。

## 仓库结构

```text
.
├── official_baseline/      # 官方baseline
├── Train/                  # 当前版本模型的训练代码
├── Infer/                  # 当前版本模型的推理代码
├── TAAC_experiments.csv    # 实验记录
└── README.md
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
| 8 | core_refactor | 模型架构引来大变，做了11天新增&消融实验的成果 | 0.8675917 | 0.253418 | 0.829255 |
| 9 | sparse_moe | token-level sparse MoE | 0.86811 | 0.254258 | 0.829527 |
| 10 | feature_tokens | 按特征类型拆分 user-dense tokenization | 0.869321 | 0.253492 | 0.829901 |
| 11 | final_no_moe | 消融实验，关闭 MoE | 0.868445 | 0.255211 | 0.830774 |

## 实验说明

### baseline

官方baseline，啥也没改。

### bf16_amp

加入BF16来提速，单轮训练时间从约67分钟降到42分钟，线上AUC小幅损失。

### recency_dense

启发于神秘EDA结果，为每个domain加入dense统计特征：

- `log1p(seq_len)`
- `log1p(last_gap)`
- missing flag
- `gap <= 60`, `gap <= 3600`, `gap <= 86400`

总共24维，通过zero-init gate融合。

### tuple_tokens

官方说user字段编号为`62/63/64/65/66`的int和dense有关联，所以试着建模为共享 tuple token。

### calendar_tokens

在序列侧额外加入calendar token，包含hour-of-day、day-of-week、day-of-month。

### time_din

仿照DIN的思路，在生成Q token的模块中将候选item与当前时间去做target aware。

### multi_scale_queries

用short/mid/long三种query view生成Q token，兼顾近期行为与长期偏好。

### time_tokens

神秘提分点，直接在user侧新增calendar token。

### core_refactor

沉寂了11天，对主线进行了大规模重构，从baseline出发重新整合已验证有效的模块：

- short/mid/long multi-scale query decomposition
- domain-specific current-time KV conditioning
- focal loss
- user dense BatchNorm
- dense-only EMA
- stronger query dropout

### sparse_moe

加入sparse MoE：每个token在8个专家中选择top-2。

### feature_tokens

按特征类型拆分 user-dense tokenization：

- 字段 `61` 和 `87`：embedding-like dense token
- 字段 `62-66`：stat-bucket dense features，先 `clamp` 再 `log1p` 后投影
- 字段 `89-91`：quantile-style feature token

### final_no_moe

少就是多，消融MoE。

---

<a id="english"></a>

# TAAC 2026 Competition Solution

[中文](#chinese)

Final score: 0.83077. We did not make it to the second round, but this is still a result we are happy with. On the `main` branch, I committed the mainline experiments in the order we actually developed them, so the history reflects the real update process. Failed experiments are not open-sourced because including them would make the repository too noisy.

## Repository Structure

```text
.
├── official_baseline/      # Official baseline
├── Train/                  # Training code for the current model
├── Infer/                  # Inference code for the current model
├── TAAC_experiments.csv    # Experiment log
└── README.md
```

The first two public commits (`baseline` and `bf16_amp`) follow the original root-level official baseline layout and do not yet split `Train/` and `Infer/`. Starting from `recency_dense`, the public code is split into three directories: `official_baseline/`, `Train/`, and `Infer/`.

## Mainline Results

| Step | Experiment | Key idea | Local AUC | Local loss | Online AUC |
|---:|---|---|---:|---:|---:|
| 0 | baseline | Official PyTorch baseline | 0.86219 | 0.22466 | 0.812646 |
| 1 | bf16_amp | BF16 mixed precision training | 0.86247 | 0.22444 | 0.81223 |
| 2 | recency_dense | Per-domain recency and length dense features | 0.8619777 | 0.224132 | 0.814566 |
| 3 | tuple_tokens | Tuple tokens for fields 62-66 | 0.864002 | 0.22274 | 0.816017 |
| 4 | calendar_tokens | Sequence calendar tokens without month-of-year | 0.8655957 | 0.221923 | 0.821676 |
| 5 | time_din | Time-conditioned DIN sequence pooling | 0.8666739 | 0.2227302 | 0.825175 |
| 6 | multi_scale_queries | Short, mid, and long query views | 0.867078 | 0.221575 | 0.827658 |
| 7 | time_tokens | Explicit current-time NS token | 0.8671137 | 0.220959 | 0.828348 |
| 8 | core_refactor | Major architecture change after 11 days of additions and ablations | 0.8675917 | 0.253418 | 0.829255 |
| 9 | sparse_moe | Token-level sparse MoE | 0.86811 | 0.254258 | 0.829527 |
| 10 | feature_tokens | User-dense tokenization by feature type | 0.869321 | 0.253492 | 0.829901 |
| 11 | final_no_moe | Ablation: disable MoE | 0.868445 | 0.255211 | 0.830774 |

## Experiment Notes

### baseline

Official baseline, unchanged.

### bf16_amp

Add BF16 for speed. One training epoch dropped from about 67 minutes to 42 minutes, with a small online AUC loss.

### recency_dense

Inspired by mysterious EDA results, add dense statistical features for each domain:

- `log1p(seq_len)`
- `log1p(last_gap)`
- missing flag
- `gap <= 60`, `gap <= 3600`, `gap <= 86400`

24 dimensions in total, fused through a zero-init gate.

### tuple_tokens

The official description says user fields `62/63/64/65/66` have related integer and dense signals, so we tried modeling them as shared tuple tokens.

### calendar_tokens

Add calendar tokens on the sequence side, including hour-of-day, day-of-week, and day-of-month.

### time_din

Following the idea of DIN, make the candidate item and current time target-aware in the module that generates Q tokens.

### multi_scale_queries

Use short/mid/long query views to generate Q tokens, covering both recent behavior and long-term preference.

### time_tokens

A mysterious score booster: directly add calendar tokens on the user side.

### core_refactor

After 11 quiet days, we made a large mainline refactor and re-integrated validated modules from the baseline:

- short/mid/long multi-scale query decomposition
- domain-specific current-time KV conditioning
- focal loss
- user dense BatchNorm
- dense-only EMA
- stronger query dropout

### sparse_moe

Add sparse MoE: each token selects the top 2 experts out of 8.

### feature_tokens

Split user-dense tokenization by feature type:

- fields `61` and `87`: embedding-like dense tokens
- fields `62-66`: stat-bucket dense features with `clamp` followed by `log1p` before projection
- fields `89-91`: quantile-style feature tokens

### final_no_moe

Less is more. Ablate MoE.

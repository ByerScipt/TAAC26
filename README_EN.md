# TAAC 2026 Competition Solution

[中文](README.md)

This repository is a cleaned open-source version of our team's TAAC 2026 competition solution. The repository keeps a clean mainline commit history: starting from the official PyTorch baseline, it includes only the experiments that were finally kept. Each commit records `Why / What changed / Impact`, along with local and online AUC.

Failed experiments are not included in the public history or experiment table. Some of them informed later design directions, especially `core_refactor`, but the repository keeps only the mainline improvements.

## Repository Structure

```text
.
├── official_baseline/      # Official PyTorch baseline kept for reference
├── Train/                  # Training code for the final mainline model
├── Infer/                  # Inference code for submission generation
├── TAAC_experiments.csv    # Public mainline experiment log
├── README.md              # Chinese README
└── README_EN.md           # English README
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
| 6 | multi_scale_queries | Short, mid, and long interest queries | 0.867078 | 0.221575 | 0.827658 |
| 7 | time_tokens | Explicit current-time context tokens | 0.8671137 | 0.220959 | 0.828348 |
| 8 | core_refactor | Core rebuild based on additional ablations | 0.8675917 | 0.253418 | 0.829255 |
| 9 | sparse_moe | Token-level sparse MoE | 0.86811 | 0.254258 | 0.829527 |
| 10 | feature_tokens | Heterogeneous feature tokenization | 0.869321 | 0.253492 | 0.829901 |
| 11 | final_no_moe | Keep heterogeneous tokenization and disable MoE by default | 0.868445 | 0.255211 | 0.830774 |

## Experiment Notes

### baseline

The official PyTorch baseline is kept as the starting point. It preserves the original schema, metric, submission format, and inference protocol.

### bf16_amp

BF16 AMP was added to improve iteration speed. It reduced one training epoch from about 67 minutes to 42 minutes, with a small online AUC loss.

### recency_dense

This step adds cheap per-domain dense statistics:

- `log1p(seq_len)`
- `log1p(last_gap)`
- missing flag
- `gap <= 60`, `gap <= 3600`, `gap <= 86400`

The 24-dimensional feature block is fused through a zero-initialized gate.

### tuple_tokens

Fields `62/63/64/65/66` contain both integer and dense signals, so this step models them as shared tuple tokens.

### calendar_tokens

Sequence timestamps are expanded into calendar tokens. The kept variant uses hour-of-day, day-of-week, and day-of-month, while month-of-year is removed after ablation.

### time_din

Current time context is introduced into DIN-style sequence pooling, allowing the model to select relevant historical events under both the candidate item and current-time conditions.

### multi_scale_queries

User interests are decoded through short, mid, and long query views, covering both recent behavior and long-term preference.

### time_tokens

Current time is injected as explicit non-sequence tokens, allowing the model to attend to time context directly.

### core_refactor

This is a larger mainline refactor based on private module-addition and ablation experiments. To keep the mainline clear, failed experiments are not described in detail.

This step starts from the baseline and re-integrates modules that had shown reliable gains:

- short/mid/long multi-scale query decomposition
- domain-specific current-time KV conditioning
- focal loss
- user dense BatchNorm
- dense-only EMA
- stronger query dropout

This was the first public mainline checkpoint to reach the `0.829+` online AUC range.

### sparse_moe

A minimal token-level sparse MoE is added: each token selects the top 2 experts out of 8, helping handle domain-level and token-level heterogeneity.

### feature_tokens

The input tokenization is split by feature type:

- fields `61` and `87` as embedding-like dense tokens
- fields `62-66` as tuple tokens
- fields `89-91` as quantile-style feature tokens

### final_no_moe

The final ablation shows that keeping heterogeneous feature tokenization while disabling sparse MoE by default works better online. MoE remains available as an optional switch for reproduction and controlled ablations.

## Training

```bash
cd Train
bash run.sh
```

Dataset paths depend on the official TAAC runtime environment. If your local paths differ, pass the required arguments through `run.sh` or adjust the data path arguments.

## Inference

```bash
cd Infer
python infer.py
```

The inference script follows the official evaluation container environment variables:

```text
MODEL_OUTPUT_PATH   checkpoint directory
EVAL_DATA_PATH      test data directory
EVAL_RESULT_PATH    output directory for predictions.json
```

#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 当前主配置：RankMixer NS tokenizer，不依赖 ns_groups.json ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --valid_num_workers 4 \
    --batch_size 256 \
    --log_every_n_steps 200 \
    --patience 3 \
    --amp_dtype bf16 \
    --use_engineered_dense_features \
    --use_shared_fid_tuple_token \
    --shared_fids 62,63,64,65,66 \
    --shared_fid_tuple_mode replace \
    "$@"

# ---- 备选配置：由 ns_groups.json 驱动的 GroupNSTokenizer ----
# 该配置使用 ns_groups.json 中的特征分组：7 个 user 组和 4 个 item 组。
# 当 d_model=64 且 num_ns=12 时，只有 num_queries=1 满足 d_model % T == 0。
# 这里 T = num_queries*4 + num_ns。切换时注释上面的主配置，并取消下面配置的注释。
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"

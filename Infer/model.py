"""PCVRHyFormer 模型，用于预测点击后的转化概率。"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, NamedTuple, Tuple, Optional, Union

LOCAL_TIME_OFFSET_SECONDS = 8 * 3600


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    timestamp: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}，S 是该序列域的 side-info 数
    seq_lens: dict        # {domain: tensor [B]}，每个样本的有效序列长度
    seq_time_buckets: dict  # {domain: tensor [B, L]}，每个事件的时间差 bucket
    seq_calendar_feats: Optional[dict] = None  # {domain: tensor [B, 3, L]} event calendar ids
    engineered_dense_feats: Optional[torch.Tensor] = None


# ═══════════════════════════════════════════════════════════════════════════════
# 旋转位置编码（RoPE）
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """预计算并缓存 RoPE 使用的 cos/sin 表。

    RoPE 在注意力中把位置信息注入 Q/K。这里提前构建最长序列需要的缓存，forward
    时只切片取用，减少运行时开销。

    属性：
        dim: RoPE 作用的维度，通常等于 head_dim。
        max_seq_len: 缓存支持的最大序列长度。
        base: 旋转位置编码的频率基数。
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 预计算频率倒数，形状为 (dim // 2,)。
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # 构建 cos/sin 缓存。
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回指定序列长度需要的 cos/sin。

        缓存在 ``__init__`` 中按 ``max_seq_len`` 一次性创建。forward 阶段只做切片和
        device 对齐，保持计算图简单。
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把最后一维前后两半交换，并对后一半取负，这是 RoPE 的旋转操作。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """把 RoPE 应用到一个 Q 或 K 张量上。

    参数：
        x: 形状 ``(B, num_heads, L, head_dim)``。
        cos: 形状 ``(1, L_max, head_dim)``；当每个样本位置不同，也可以是
            ``(B, L, head_dim)``。
        sin: 和 ``cos`` 同形状。

    返回：
        旋转后的张量，形状仍为 ``(B, num_heads, L, head_dim)``。
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer 基础组件
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU 前馈层，核心激活为 ``x1 * SiLU(x2)``。"""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """支持 RoPE 的多头注意力。

    这里手动完成 Q/K/V 线性投影和多头 reshape，在点积注意力前把 RoPE 加到 Q/K
    上。实际注意力计算使用 PyTorch 的 ``scaled_dot_product_attention``。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """计算多头注意力，并按需加入 RoPE。

        参数：
            query: ``(B, Lq, D)``。
            key: ``(B, Lk, D)``。
            value: ``(B, Lk, D)``。
            key_padding_mask: ``(B, Lk)``，True 表示 padding 位置。
            attn_mask: ``(Lq, Lk)`` 或 ``(B*num_heads, Lq, Lk)``，additive mask。
            rope_cos: ``(1, L, head_dim)``，KV 侧 RoPE；未传入 q_rope_* 时也用于 Q。
            rope_sin: 和 ``rope_cos`` 同形状。
            q_rope_cos: ``(B, Lq, head_dim)`` 或 ``(1, Lq, head_dim)``，Q 侧专用
                RoPE，常用于 cross-attention 中按原始位置 gather 出来的 query。
            q_rope_sin: 和 ``q_rope_cos`` 同形状。
            need_weights: 兼容旧接口的参数，当前不使用。

        返回：
            ``(output, None)``。
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. 线性投影到 Q/K/V。
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. 拆成多头格式：(B, num_heads, L, head_dim)。
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 分别给 Q 和 K 注入 RoPE。
        if rope_cos is not None and rope_sin is not None:
            # K 使用 KV 侧的位置编码。
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q 侧优先使用专门传入的位置编码，例如 LongerEncoder cross-attn 中
                # top_k token 对应的原始位置。
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. 将 padding mask 转成 SDPA 接受的 bool mask。
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk)，True 表示 padding。
            # SDPA mask 中 True 表示允许 attend，所以这里需要取反。
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask 是 additive float mask，0 表示可见，-inf 表示不可见。
            # 转成 bool mask 后与 padding mask 合并。
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. 执行 scaled dot-product attention。
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # 全 padding softmax 会产生 NaN。这里置零，残差连接会保留原始输入路径。
        out = torch.nan_to_num(out, nan=0.0)

        # 6. 合并多头并做输出投影。
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Q token 到序列 token 的 cross-attention。

    Query 来自全局 Q token，Key/Value 来自某一路序列 token。这里设置
    ``rope_on_q=False``，只给序列侧 K 注入位置编码。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算 Q token 对序列 token 的 cross-attention。

        参数：
            query: ``(B, Nq, D)``，当前序列域对应的 query token。
            key_value: ``(B, L, D)``，该序列域的 event token。
            key_padding_mask: ``(B, L)``，True 表示 padding 位置。
            rope_cos: ``(1, L, head_dim)``，KV 侧 RoPE cos。
            rope_sin: ``(1, L, head_dim)``，KV 侧 RoPE sin。

        返回：
            更新后的 query token，形状 ``(B, Nq, D)``。
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer 的 Query Boosting 模块。

    输入是所有序列的 decoded Q token 加上 NS token，形状为 ``(B, T, D)``。
    full 模式下会先通过无参数 reshape/transpose 做 token mixing，再经过共享 FFN，
    最后和原输入做残差连接。

    约束：``full`` 模式要求 ``d_model`` 能被 ``T`` 整除。
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq*S + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # 纯恒等映射，不创建子模块。
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # 每个 token 共享同一套 FFN，full 和 ffn_only 模式都会使用。
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # 残差后加 LayerNorm，稳定多层 block 堆叠。
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """通过 reshape 和 transpose 做无参数 token mixing。

        步骤：
        1. 把通道切成 T 个子空间：``(B, T, D) -> (B, T, T, d_sub)``。
        2. 交换 token 轴和子空间轴：``(B, token, h, d_sub) -> (B, h, token, d_sub)``。
        3. 展平回 ``(B, T, D)``。

        参数：
            Q: ``(B, T, D)``。

        返回：
            mixing 后的 ``(B, T, D)``。
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """执行 Query Boosting。

        参数：
            Q: ``(B, T, D)``，其中 ``T = Nq*S + Nns``。

        返回：
            增强后的 token，形状 ``(B, T, D)``。
        """
        if self.mode == 'none':
            return Q

        # full 模式做无参数 token mixing，ffn_only 模式直接进入 FFN。
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # 对每个 token 应用共享 FFN。
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # 和原始 Q 做残差连接。
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """多序列 query token 生成模块。

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, DINPool(Seq_i, candidate_item))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]

    这样每一路序列都有专属 query，但 query 的生成会看到共享的 user/item NS 信息。
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_item_ns: int,
        has_item_dense: bool,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        self.num_item_ns = num_item_ns
        self.has_item_dense = has_item_dense
        self.num_item_side_tokens = num_item_ns + (1 if has_item_dense else 0)

        global_info_dim = (num_ns + 1) * d_model

        # 对拼接后的 global_info 做 LayerNorm，稳定大维度拼接后的数值尺度。
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Compress all item-side NS tokens (item_ns + optional item_dense)
        # into one d_model query for DIN target-aware pooling.
        item_concat_dim = self.num_item_side_tokens * d_model
        self.item_query_norm = nn.LayerNorm(item_concat_dim)
        self.item_query_mlp = nn.Sequential(
            nn.Linear(item_concat_dim, d_model * hidden_mult),
            nn.SiLU(),
            nn.Linear(d_model * hidden_mult, d_model),
            nn.LayerNorm(d_model),
        )

        # DIN-style target-aware pooling:
        # score_t = MLP([q, k_t, q-k_t, q*k_t]), then masked softmax weighted sum.
        din_hidden_dim = d_model * hidden_mult
        self.din_attn_mlp = nn.Sequential(
            nn.Linear(4 * d_model, din_hidden_dim),
            nn.SiLU(),
            nn.Linear(din_hidden_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

        # 每个序列域拥有 N 个独立 FFN，分别生成 N 个 query token。
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        time_context: Optional[torch.Tensor] = None,
    ) -> list:
        """为每个序列域生成 query token。

        参数：
            ns_tokens: ``(B, M, D)``，共享 NS token。
            seq_tokens_list: 长度为 S 的列表，每项形状 ``(B, L_i, D)``。
            seq_padding_masks: 长度为 S 的列表，每项形状 ``(B, L_i)``，True 表示
                padding。

        返回：
            长度为 S 的 query token 列表，每项形状 ``(B, Nq, D)``。
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        # Candidate-item conditioning vector: concat all item-side tokens
        # (item_ns + optional item_dense), then project to d_model.
        item_side_tokens = ns_tokens[:, -self.num_item_side_tokens:, :]  # (B, K_item, D)
        item_concat = item_side_tokens.reshape(B, -1)  # (B, K_item*D)
        item_concat = self.item_query_norm(item_concat)
        candidate_item = self.item_query_mlp(item_concat)  # (B, D)
        if time_context is not None:
            candidate_item = candidate_item + time_context

        q_tokens_list = []
        for i in range(self.num_sequences):
            # DIN-style target-aware weighted pooling.
            seq_tokens = seq_tokens_list[i]  # (B, L_i, D)
            valid_mask = ~seq_padding_masks[i]  # (B, L_i), True = valid

            L_i = seq_tokens.shape[1]
            query = candidate_item.unsqueeze(1).expand(-1, L_i, -1)  # (B, L_i, D)
            attn_in = torch.cat(
                [query, seq_tokens, query - seq_tokens, query * seq_tokens],
                dim=-1,
            )  # (B, L_i, 4D)
            attn_logits = self.din_attn_mlp(attn_in).squeeze(-1)  # (B, L_i)
            attn_logits = attn_logits.masked_fill(~valid_mask, -1e9)
            attn_weights = F.softmax(attn_logits, dim=1)  # (B, L_i)
            attn_weights = attn_weights * valid_mask.float()
            weight_denom = attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            attn_weights = attn_weights / weight_denom
            seq_pooled = torch.bmm(
                attn_weights.unsqueeze(1),
                seq_tokens,
            ).squeeze(1)  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)。
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # 生成 N 个 query token。
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# 序列编码器
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """轻量的无注意力序列编码器。

    结构为 ``x + Dropout(SwiGLU(LN(x)))``。它只做逐 token 的非线性变换，不在序列
    token 之间建模注意力关系。
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """应用带残差的 SwiGLU 编码。

        参数：
            x: ``(B, L, D)``。
            key_padding_mask: ``(B, L)``，True 表示 padding。该编码器不使用它。
            **kwargs: 接收 rope_cos/rope_sin 等未使用参数，保持统一接口。

        返回：
            ``(output, key_padding_mask)``，其中 output 形状为 ``(B, L, D)``。
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """带 self-attention 和 RoPE 的高容量序列编码器。

    结构是标准 Pre-LN Transformer Encoder Layer。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """执行一层 Transformer encoder。

        参数：
            x: ``(B, L, D)``。
            key_padding_mask: ``(B, L)``，True 表示 padding。
            rope_cos: ``(1, L, head_dim)``，RoPE cos。
            rope_sin: ``(1, L, head_dim)``，RoPE sin。

        返回：
            ``(output, key_padding_mask)``，其中 output 形状为 ``(B, L, D)``。
        """
        # Pre-LN self-attention，可选 RoPE。
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # Pre-LN FFN。
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K 压缩序列编码器。

    根据输入长度选择行为：
    - ``L > top_k`` 时，取最近的 top_k 个有效 token 作为 Q，完整序列作为 K/V，
      通过 cross-attention 压缩成 ``(B, top_k, D)``。
    - ``L <= top_k`` 时，在当前 token 上做 self-attention，输出仍为
      ``(B, top_k, D)`` 或当前长度对应形状。

    causal mask 只作用在 top_k token 的 self-attention 阶段。第一层 cross-attention
    中 Q 和 K 长度不同，直接让最近 top_k token attend 完整历史。

    返回 ``(output, new_key_padding_mask)``，下游 block 会继续使用更新后的 mask。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # attention 前的 LayerNorm。
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # cross-attention 和 self-attention 共用这套 RoPE MHA。
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # Pre-LN FFN 加残差。
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为每个样本选出最近的 top_k 个有效 token。

        参数：
            x: ``(B, L, D)``。
            key_padding_mask: ``(B, L)``，True 表示 padding。

        返回：
            top_k_tokens: ``(B, top_k, D)``。
            new_padding_mask: ``(B, top_k)``，True 表示 padding。
            position_indices: ``(B, top_k)``，所选 token 在原序列中的位置，用于
                Q 侧 RoPE。
        """
        B, L, D = x.shape
        device = x.device

        # 每个样本的有效长度。
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # 每个样本的起始位置：max(valid_len - top_k, 0)。
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # 构造 gather 索引，形状 (B, top_k)。
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # 有效长度小于 top_k 时，前部位置由 mask 标成 padding。索引本身先裁剪到
        # [0, L-1]，保证 gather 合法。
        indices = torch.clamp(indices, min=0, max=L - 1)

        # gather 得到 (B, top_k, D)。
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # 新 mask 中，前 (top_k - actual_k) 个位置是 padding。
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # padding 位置清零。
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # Q 侧 RoPE 使用这些原始位置索引。
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行 LongerEncoder 的自适应 cross/self attention。

        参数：
            x: ``(B, L, D)``，序列 token。
            key_padding_mask: ``(B, L)``，True 表示 padding。
            rope_cos: ``(1, L, head_dim)``，长度覆盖原始序列长度 L。
            rope_sin: ``(1, L, head_dim)``。

        返回：
            output: ``(B, top_k, D)``，压缩后的序列。
            new_key_padding_mask: ``(B, top_k)``，更新后的 padding mask。
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention 模式，通常出现在第一个 MultiSeqHyFormerBlock ===
            # 1. 取最近 top_k token 作为 query。
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. attention 前归一化。
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. 从全局 RoPE 表中按 top_k 原始位置取出 Q 侧 cos/sin。
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim)，q_pos_indices: (B, top_k)。
                head_dim = rope_cos.shape[2]
                # 扩展到 batch 维，便于按样本 gather。
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. cross-attention。Q 和 K 长度不同，此处不使用 causal mask。
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # 原始 (B, L) mask。
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # 基于 q 的残差。
        else:
            # === Self Attention 模式，用于后续已经压缩过的序列 ===
            new_mask = key_padding_mask

            # Pre-LN，Q/K/V 共用 norm_q。
            x_normed = self.norm_q(x)

            # 可选 causal mask。
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # Pre-LN FFN 加残差。
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """按配置创建序列编码器。

    参数：
        encoder_type: ``'swiglu'``、``'transformer'`` 或 ``'longer'``。
        d_model: 模型隐藏维度。
        num_heads: 注意力头数，transformer/longer 使用。
        hidden_mult: FFN 扩展倍数。
        dropout: dropout 比例。
        top_k: LongerEncoder 的压缩长度。
        causal: LongerEncoder 是否启用 causal mask。

    返回：
        序列编码器模块。
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Block
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """多序列 HyFormer block。

    每一路序列先独立完成 Sequence Evolution 和 Query Decoding。随后把所有序列域的
    Q token 与共享 NS token 拼在一起，通过 RankMixer 做联合 Query Boosting。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full'
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # 每个序列域一套独立 sequence encoder。
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # 每个序列域一套独立 cross-attention。
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer 的输入 token 数：Nq * S + Nns。
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """执行一个 MultiSeqHyFormerBlock。

        参数：
            q_tokens_list: 长度为 S 的列表，每项 ``(B, Nq, D)``。
            ns_tokens: ``(B, Nns, D)``。
            seq_tokens_list: 长度为 S 的列表，每项 ``(B, L_i, D)``。
            seq_padding_masks: 长度为 S 的列表，每项 ``(B, L_i)``。
            rope_cos_list: 长度为 S 的 RoPE cos 列表。
            rope_sin_list: 长度为 S 的 RoPE sin 列表。

        返回：
            ``(next_q_list, next_ns, next_seq_list, next_masks)``。其中
            next_q_list 是更新后的各序列 query，next_ns 是更新后的 NS token，
            next_seq_list 是编码后的各序列 token，next_masks 是对应 padding mask。
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. 各序列域独立做 Sequence Evolution。
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. 各序列域独立做 Query Decoding：Q attend 自己对应的序列。
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion：拼接所有 decoded Q 和 NS token。
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting：RankMixer 在 Q/NS token 之间做交叉。
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. 切回各序列域 Q token 和共享 NS token。
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer 主模型
# ═══════════════════════════════════════════════════════════════════════════════


def _filter_feature_groups(
    groups: List[List[int]],
    exclude_feature_indices: Optional[List[int]],
) -> List[List[int]]:
    if not exclude_feature_indices:
        return groups
    excluded = set(exclude_feature_indices)
    return [[idx for idx in group if idx not in excluded] for group in groups]


class GroupNSTokenizer(nn.Module):
    """``ns_tokenizer_type='group'`` 时使用的 NS tokenizer。

    它先按 feature group 收集离散特征，对单值特征直接查 Embedding，对多值特征查
    Embedding 后做 padding-aware mean pooling。每个 group 最后投影成一个 NS token。
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0,
                 exclude_feature_indices: Optional[List[int]] = None) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = _filter_feature_groups(groups, exclude_feature_indices)
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold
        self.exclude_feature_indices = set(exclude_feature_indices or [])

        # 每个 fid 一张 Embedding 表。被 emb_skip_threshold 跳过或缺少词表信息时记为 None。
        embs = []
        for fid_idx, (vs, offset, length) in enumerate(feature_specs):
            skip = (
                fid_idx in self.exclude_feature_indices
                or int(vs) <= 0
                or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            )
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # fid index 到 self.embs 实际下标的映射；-1 表示该特征被过滤。
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # 每个 group 一套投影：num_fids_in_group * emb_dim -> d_model。
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max(1, len(group) * emb_dim), d_model),
                nn.LayerNorm(d_model),
            )
            for group in self.groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """把分组离散特征编码成 NS token。

        参数：
            int_feats: ``(B, total_int_dim)``，拼接后的离散特征张量。

        返回：
            ``(B, num_groups, D)``。
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # 被过滤的高基数特征使用零向量占位。
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # 单值特征直接查表。
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # 多值特征查表后对非 padding 位置做 mean pooling。
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            if fid_embs:
                cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            else:
                cat_emb = int_feats.new_zeros(int_feats.shape[0], 1, dtype=torch.float)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """RankMixer 风格的 NS tokenizer。

    它会按 group 顺序把所有离散特征的 Embedding 拼成一个长向量，再平均切成
    ``num_ns_tokens`` 段，每段投影成一个 ``d_model`` 维 token。这样 NS token 数可以
    独立调节，不再等于 group 数。
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
        exclude_feature_indices: Optional[List[int]] = None,
    ) -> None:
        """初始化 RankMixerNSTokenizer。

        参数：
            feature_specs: 每个特征的 ``(vocab_size, offset, length)``。
            groups: feature index 分组，同时决定拼接顺序。
            emb_dim: 单个特征 Embedding 维度。
            d_model: 输出 token 维度。
            num_ns_tokens: 需要产出的 NS token 数。
            emb_skip_threshold: 词表大小超过该阈值的特征不创建 Embedding。
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = [group for group in _filter_feature_groups(groups, exclude_feature_indices) if group]
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold
        self.exclude_feature_indices = set(exclude_feature_indices or [])
        if num_ns_tokens <= 0:
            raise ValueError(f"num_ns_tokens must be positive, got {num_ns_tokens}")

        # 每个 fid 一张 Embedding 表。被阈值过滤或缺少词表信息时记为 None。
        embs = []
        for fid_idx, (vs, offset, length) in enumerate(feature_specs):
            skip = (
                fid_idx in self.exclude_feature_indices
                or int(vs) <= 0
                or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            )
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # fid index 到 self.embs 实际下标的映射；-1 表示该特征被过滤。
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # 计算拼接后的总维度：所有有效 fid 数 * emb_dim。
        total_num_fids = sum(len(g) for g in self.groups)
        if total_num_fids == 0:
            raise ValueError("RankMixerNSTokenizer has no active features after exclusions")
        total_emb_dim = total_num_fids * emb_dim

        # 通过尾部补零让总维度可以被 num_ns_tokens 整除。
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # 每个 chunk 单独投影到 d_model，并做 LayerNorm。
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """查表、拼接、切块并投影成 NS token。

        参数：
            int_feats: ``(B, total_int_dim)``，拼接后的离散特征张量。

        返回：
            ``(B, num_ns_tokens, d_model)``。
        """
        # 1. 按 group 顺序查所有 fid 的 Embedding，然后拼成一个长向量。
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. 需要时在末尾补零。
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. 切成 num_ns_tokens 个 chunk，每个 chunk 投影成一个 token。
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # 每项形状为 (B, chunk_dim)。
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class SharedFidTupleTokenizer(nn.Module):
    """把共享 fid 的 user_int 和 user_dense 对齐编码成一个 tuple token。

    这个模块服务当前 best 中的 tokenization 改动：对于同一组 fid，int 部分提供离散
    ID，dense 部分提供与该 ID 对齐的数值。逐元素结合后再聚合，避免二者分别进入
    generic token 后丢失对齐关系。
    """

    def __init__(
        self,
        tuple_specs: List[Dict[str, int]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        if not tuple_specs:
            raise ValueError("SharedFidTupleTokenizer requires at least one tuple spec")
        self.tuple_specs = tuple_specs
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.emb_skip_threshold = emb_skip_threshold
        self.fids = [int(spec['fid']) for spec in tuple_specs]
        self.vocab_sizes = [int(spec['int_vocab_size']) for spec in tuple_specs]

        id_embs = []
        self._emb_index = []
        for vocab_size in self.vocab_sizes:
            skip = int(vocab_size) <= 0 or (
                emb_skip_threshold > 0 and int(vocab_size) > emb_skip_threshold
            )
            if skip:
                self._emb_index.append(-1)
            else:
                self._emb_index.append(len(id_embs))
                id_embs.append(nn.Embedding(int(vocab_size) + 1, emb_dim, padding_idx=0))
        self.id_embs = nn.ModuleList(id_embs)

        self.value_projs = nn.ModuleList([
            nn.Linear(1, emb_dim)
            for _ in tuple_specs
        ])
        self.tuple_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim * 2, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in tuple_specs
        ])
        self.aggregate_proj = nn.Sequential(
            nn.Linear(len(tuple_specs) * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self._logged_sanity = False

    def forward(
        self,
        user_int_feats: torch.Tensor,
        user_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        fid_tokens = []
        valid_any = []
        valid_rates = {}
        stat_values = []

        for idx, spec in enumerate(self.tuple_specs):
            fid = int(spec['fid'])
            int_offset = int(spec['int_offset'])
            dense_offset = int(spec['dense_offset'])
            length = int(spec['int_length'])
            dense_length = int(spec['dense_length'])
            if length != dense_length:
                raise ValueError(
                    f"shared fid {fid} int_length={length} != dense_length={dense_length}"
                )

            int_ids = user_int_feats[:, int_offset:int_offset + length].long()
            dense_vals = user_dense_feats[:, dense_offset:dense_offset + length].float()
            finite_mask = torch.isfinite(dense_vals)
            mask = (int_ids > 0) & finite_mask

            value = torch.sign(dense_vals) * torch.log1p(torch.abs(dense_vals))
            value = torch.clamp(value, min=-20.0, max=20.0)
            value = torch.where(finite_mask, value, torch.zeros_like(value))

            emb_real_idx = self._emb_index[idx]
            if emb_real_idx == -1:
                id_emb = user_int_feats.new_zeros(
                    int_ids.shape[0], int_ids.shape[1], self.emb_dim, dtype=torch.float
                )
            else:
                id_emb = self.id_embs[emb_real_idx](int_ids)
            value_emb = self.value_projs[idx](value.unsqueeze(-1))
            tuple_emb = F.silu(self.tuple_projs[idx](torch.cat([id_emb, value_emb], dim=-1)))

            mask_f = mask.unsqueeze(-1).to(tuple_emb.dtype)
            tuple_emb = tuple_emb * mask_f
            count = mask_f.sum(dim=1).clamp(min=1.0)
            fid_tokens.append(tuple_emb.sum(dim=1) / count)
            valid_any.append(mask.any(dim=1, keepdim=True))

            if not self._logged_sanity:
                valid_rates[fid] = float(mask.float().mean().detach().cpu().item())
                if mask.any():
                    stat_values.append(value[mask].detach().float())

        cat = torch.cat(fid_tokens, dim=-1)
        shared_token = F.silu(self.aggregate_proj(cat)).unsqueeze(1)
        any_valid = torch.stack(valid_any, dim=0).any(dim=0).to(shared_token.dtype).unsqueeze(-1)
        shared_token = shared_token * any_valid

        if not self._logged_sanity:
            if stat_values:
                stats = torch.cat(stat_values, dim=0)
            else:
                stats = shared_token.new_zeros(1).detach().float()
            nan_count = int(torch.isnan(stats).sum().item())
            inf_count = int(torch.isinf(stats).sum().item())
            logging.info(
                "shared_fid_tuple_token sanity: "
                f"shape={tuple(shared_token.shape)} "
                f"tuple_valid_rate_per_fid={valid_rates} "
                f"transformed_dense_min={stats.min().item():.6f} "
                f"transformed_dense_max={stats.max().item():.6f} "
                f"transformed_dense_mean={stats.mean().item():.6f} "
                f"nan_count={nan_count} inf_count={inf_count}"
            )
            self._logged_sanity = True

        return shared_token


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer 主模型。

    输入先被拆成三类 token：
    - NS token：来自 user/item 离散特征、dense 特征，以及可选 tuple token。
    - sequence token：每个 seq_a/seq_b/seq_c/seq_d 事件一枚 token。
    - query token：由 NS token 和每路序列的 pooled summary 生成。

    之后每个 HyFormer block 先让 query attend 对应序列，再把所有 query 和 NS token
    放进 RankMixer 交叉，最后只取 query 输出做分类。
    """

    def __init__(
        self,
        # 数据 schema。
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS 分组配置，内部使用 fid index。
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # 模型超参数。
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        use_seq_calendar_features: bool = True,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer 变体。
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        use_engineered_dense_features: bool = True,
        engineered_dense_dim: int = 24,
        use_shared_fid_tuple_token: bool = False,
        shared_fids: Optional[Union[str, List[int]]] = None,
        shared_fid_tuple_mode: str = 'replace',
        shared_fid_tuple_specs: Optional[List[Dict[str, int]]] = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # 固定顺序，保证训练和推理一致。
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.use_seq_calendar_features = use_seq_calendar_features
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.use_engineered_dense_features = use_engineered_dense_features
        self.engineered_dense_dim = engineered_dense_dim if use_engineered_dense_features else 0
        self.use_shared_fid_tuple_token = use_shared_fid_tuple_token
        self.shared_fid_tuple_mode = shared_fid_tuple_mode
        if shared_fid_tuple_mode not in ('replace', 'additive'):
            raise ValueError(
                f"shared_fid_tuple_mode must be replace/additive, got {shared_fid_tuple_mode}"
            )
        self.shared_fids = (
            [int(fid) for fid in shared_fids.split(',') if fid.strip()]
            if isinstance(shared_fids, str)
            else [int(fid) for fid in (shared_fids or [])]
        )
        self.shared_fid_tuple_specs = shared_fid_tuple_specs or []
        self.shared_fid_tuple_token_count = 1 if use_shared_fid_tuple_token else 0
        if self.use_shared_fid_tuple_token and not self.shared_fid_tuple_specs:
            raise ValueError("shared_fid_tuple_specs is required when use_shared_fid_tuple_token=True")
        shared_int_feature_indices = (
            [int(spec['int_feature_idx']) for spec in self.shared_fid_tuple_specs]
            if self.use_shared_fid_tuple_token and shared_fid_tuple_mode == 'replace'
            else []
        )
        shared_dense_slices = (
            [(int(spec['dense_offset']), int(spec['dense_length'])) for spec in self.shared_fid_tuple_specs]
            if self.use_shared_fid_tuple_token and shared_fid_tuple_mode == 'replace'
            else []
        )
        self.original_shared_fids_still_in_generic_tokens = (
            bool(self.use_shared_fid_tuple_token)
            and shared_fid_tuple_mode == 'additive'
        )

        if self.use_shared_fid_tuple_token:
            self.shared_fid_tuple_tokenizer = SharedFidTupleTokenizer(
                tuple_specs=self.shared_fid_tuple_specs,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
        else:
            self.shared_fid_tuple_tokenizer = None
        logging.info(f"use_shared_fid_tuple_token={self.use_shared_fid_tuple_token}")
        logging.info(f"shared_fids={self.shared_fids}")
        logging.info(f"shared_fid_tuple_mode={self.shared_fid_tuple_mode}")
        logging.info(
            "original_shared_fids_still_in_generic_tokens="
            f"{self.original_shared_fids_still_in_generic_tokens}"
        )
        logging.info(f"tuple token count={self.shared_fid_tuple_token_count}")

        # ================== NS token 构造 ==================

        if ns_tokenizer_type == 'group':
            # group 模式：每个分组产出一个 NS token。
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
                exclude_feature_indices=shared_int_feature_indices,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # rankmixer 模式：所有 Embedding 先拼接，再切块投影成固定数量 token。
            # token 数为 0 时自动回退到 group 数。
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            generic_user_ns_tokens = user_ns_tokens
            if self.use_shared_fid_tuple_token and shared_fid_tuple_mode == 'replace':
                generic_user_ns_tokens = max(
                    1, user_ns_tokens - self.shared_fid_tuple_token_count)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=generic_user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                exclude_feature_indices=shared_int_feature_indices,
            )
            num_user_ns = generic_user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # user dense 整体投影成一个 NS token。
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )
            if shared_dense_slices:
                mask = torch.ones(user_dense_dim, dtype=torch.float32)
                for offset, length in shared_dense_slices:
                    mask[offset:offset + length] = 0.0
                self.register_buffer('user_dense_generic_mask', mask, persistent=False)
            else:
                self.user_dense_generic_mask = None
        else:
            self.user_dense_generic_mask = None

        # item dense 整体投影成一个 NS token。
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # 统计最终 NS token 数，用于后续 query generator 和 RankMixer。
        self.num_ns = (num_user_ns + self.shared_fid_tuple_token_count
                       + (1 if self.has_user_dense else 0)
                       + num_item_ns + (1 if self.has_item_dense else 0))

        # ================== 检查 RankMixer full 模式的维度约束 ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== 序列 token Embedding ==================
        # seq_id_threshold 决定序列 tokenizer 内哪些特征按 ID 特征处理。ID 特征训练时
        # 会使用额外 dropout。它和 emb_skip_threshold 相互独立；后者控制是否创建
        # Embedding 表。
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """创建序列 side-info 的 Embedding 列表。

            被 emb_skip_threshold 过滤或缺少词表信息（vs<=0）的特征返回 None。
            """
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # position index 到 module_list 实际下标的映射；-1 表示该特征被跳过。
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== 动态序列 Embedding：每个序列域一套表和投影 ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map。
        self._seq_is_id = {}        # domain -> is_id list。
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list。
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== 时间差 bucket Embedding，可选 ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        if self.use_seq_calendar_features:
            self.seq_calendar_embeddings = nn.ModuleDict({
                'hour_of_day': nn.Embedding(25, d_model, padding_idx=0),
                'day_of_week': nn.Embedding(8, d_model, padding_idx=0),
                'day_of_month': nn.Embedding(32, d_model, padding_idx=0),
            })
            self.seq_calendar_alpha = nn.Parameter(torch.zeros(1))

        # ================== HyFormer 组件 ==================
        # MultiSeqQueryGenerator 根据 NS token 和序列 summary 生成每路 query token。
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_item_ns=num_item_ns,
            has_item_dense=self.has_item_dense,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # 多层 MultiSeqHyFormerBlock 堆叠，负责 sequence evolution、query decoding 和 token mixing。
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # 输出投影：拼接所有序列域的最终 Q token 后压回 d_model。
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Embedding/token dropout。
        self.emb_dropout = nn.Dropout(dropout_rate)

        # 二分类分类头。
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # 初始化 Embedding 参数。
        self._init_params()

        if self.use_engineered_dense_features:
            self.engineered_proj = nn.Sequential(
                nn.LayerNorm(engineered_dense_dim),
                nn.Linear(engineered_dense_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.engineered_alpha = nn.Parameter(torch.zeros(1))

        # 记录 emb_skip_threshold 实际过滤了哪些特征。
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """对所有 Embedding 权重做 Xavier 初始化，并保持 padding 行为 0。"""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.shared_fid_tuple_tokenizer is not None:
            for emb in self.shared_fid_tuple_tokenizer.id_embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

        if self.use_seq_calendar_features:
            for emb in self.seq_calendar_embeddings.values():
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """只重置高基数 Embedding。

        低基数 Embedding 和时间特征 Embedding 会保留。trainer.py 会在重建 Adagrad
        优化器时保留这些未重置参数的优化器状态。

        参数：
            cardinality_threshold: vocab_size 大于该值的 Embedding 会被重置。

        返回：
            被重置参数的 ``data_ptr()`` 集合。
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # 已被 emb_skip_threshold 跳过，没有 Embedding 需要重置。
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        if self.shared_fid_tuple_tokenizer is not None:
            tokenizer = self.shared_fid_tuple_tokenizer
            for i, vs in enumerate(tokenizer.vocab_sizes):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.id_embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding 始终保留。
        if self.num_time_buckets > 0:
            skip_count += 1
        if self.use_seq_calendar_features:
            skip_count += len(self.seq_calendar_embeddings)

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """返回所有 Embedding 参数，trainer.py 会用 Adagrad 优化它们。"""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """返回所有非 Embedding 参数，trainer.py 会用 AdamW 优化它们。"""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        calendar_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """把某个序列域从原始 side-info ID 编码成 event token。

        输入 ``seq`` 形状是 ``[B, S, L]``，S 是该序列域的 side-info 特征数。每个
        side-info 特征先独立查 Embedding，随后在最后一维拼接，得到
        ``[B, L, S*emb_dim]``，再投影成 ``[B, L, D]``。如果启用 time bucket，
        每个事件 token 会额外加上对应的时间差 Embedding。
        """
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # 被过滤的序列 side-info 特征使用零向量占位，保持拼接维度不变。
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # 加上时间差 bucket Embedding；padding id=0 会产生零向量。
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        if self.use_seq_calendar_features and calendar_feats is not None:
            calendar_emb = (
                self.seq_calendar_embeddings['hour_of_day'](calendar_feats[:, 0, :])
                + self.seq_calendar_embeddings['day_of_week'](calendar_feats[:, 1, :])
                + self.seq_calendar_embeddings['day_of_month'](calendar_feats[:, 2, :])
            )
            token_emb = token_emb + self.seq_calendar_alpha * calendar_emb

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """根据有效序列长度生成 padding mask。"""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """执行多层 HyFormer block，并把最终 Q token 聚合成一个样本向量。"""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # 每层 block 前为当前各序列长度准备 RoPE cos/sin。
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        # 输出阶段只使用各序列域的 Q token：先拼接，再投影回 d_model。
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def _apply_engineered_dense(
        self,
        output: torch.Tensor,
        engineered_dense_feats: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_engineered_dense_features:
            return output
        if engineered_dense_feats is None:
            raise ValueError("engineered_dense_feats is required when use_engineered_dense_features=True")
        eng = self.engineered_proj(engineered_dense_feats.float())
        return output + self.engineered_alpha * eng

    def _build_ns_tokens(self, inputs: ModelInput) -> torch.Tensor:
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.shared_fid_tuple_tokenizer is not None:
            ns_parts.append(self.shared_fid_tuple_tokenizer(
                inputs.user_int_feats,
                inputs.user_dense_feats,
            ))
        if self.has_user_dense:
            user_dense_feats = inputs.user_dense_feats
            if self.user_dense_generic_mask is not None:
                user_dense_feats = user_dense_feats * self.user_dense_generic_mask.to(
                    device=user_dense_feats.device,
                    dtype=user_dense_feats.dtype,
                )
            user_dense_tok = F.silu(self.user_dense_proj(user_dense_feats)).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        return torch.cat(ns_parts, dim=1)

    def _build_time_context(self, inputs: ModelInput) -> torch.Tensor:
        """Build click-time context vector for time-conditioned DIN query."""
        if not self.use_seq_calendar_features:
            return torch.zeros(
                (inputs.user_int_feats.shape[0], self.d_model),
                device=inputs.user_int_feats.device,
            )

        timestamps = inputs.timestamp.to(device=inputs.user_int_feats.device)
        local_seconds = timestamps + LOCAL_TIME_OFFSET_SECONDS
        days = local_seconds // 86400
        seconds_in_day = local_seconds % 86400

        hour = seconds_in_day // 3600
        day_of_week = (days + 3) % 7

        local_np = local_seconds.to(torch.int64).cpu().numpy()
        months = local_np.astype('datetime64[s]').astype('datetime64[M]')
        day_of_month = (
            local_np.astype('datetime64[s]').astype('datetime64[D]')
            - months.astype('datetime64[D]')
        ).astype('int64')

        hour_ids = hour.to(torch.int64) + 1
        dow_ids = day_of_week.to(torch.int64) + 1
        dom_ids = torch.from_numpy(day_of_month + 1).to(timestamps.device)

        calendar_emb = (
            self.seq_calendar_embeddings['hour_of_day'](hour_ids)
            + self.seq_calendar_embeddings['day_of_week'](dow_ids)
            + self.seq_calendar_embeddings['day_of_month'](dom_ids)
        )
        return calendar_emb

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """执行训练阶段 forward，返回 logits。"""
        # 1. 构造 NS token：user/item generic token、可选 tuple token、dense token。
        ns_tokens = self._build_ns_tokens(inputs)

        # 2. 将每个序列域的 [B, S, L] side-info ID 编码成 [B, L, D] event token。
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_calendar_feats[domain] if inputs.seq_calendar_feats is not None else None)
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. 每个序列域根据 NS token 和自身 summary 生成独立 Q token。
        time_context = self._build_time_context(inputs)
        q_tokens_list = self.query_generator(
            ns_tokens,
            seq_tokens_list,
            seq_masks_list,
            time_context=time_context,
        )

        # 4. 多层 HyFormer block 中完成 query-sequence 交互和 Q/NS token 交叉。
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training
        )
        output = self._apply_engineered_dense(output, inputs.engineered_dense_feats)

        # 5. 分类头输出 logit。
        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行推理路径，关闭 dropout，并同时返回 logits 和最终 embedding。"""
        # 和 forward 使用同一条主路径，只是在 block stack 中关闭 dropout。
        ns_tokens = self._build_ns_tokens(inputs)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_calendar_feats[domain] if inputs.seq_calendar_feats is not None else None)
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        time_context = self._build_time_context(inputs)
        q_tokens_list = self.query_generator(
            ns_tokens,
            seq_tokens_list,
            seq_masks_list,
            time_context=time_context,
        )

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False
        )
        output = self._apply_engineered_dense(output, inputs.engineered_dense_feats)

        logits = self.clsfier(output)
        return logits, output

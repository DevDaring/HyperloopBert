"""
common/attention.py

Custom self-attention for bidirectional MLM pre-training.
Implements FlashAttention-2 (varlen/padding-aware path) with automatic
fallback to PyTorch SDPA, then eager attention.

Fallback chain:  FlashAttention-2 (varlen)  ->  SDPA  ->  Eager

BF16 is used throughout; FP16 is never requested.
The active path is logged ONCE per model build via a module-level flag.
"""

# CITATION: Dao, T. et al. (2022/2023). FlashAttention: Fast and Memory-Efficient
#           Exact Attention with IO-Awareness. NeurIPS 2022 / ICLR 2023.
#           https://arxiv.org/abs/2205.14135
# CITATION: Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional
#           Transformers for Language Understanding. NAACL 2019.
#           https://arxiv.org/abs/1810.04805  [bidirectional MLM attention design]

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import: FlashAttention-2
# ---------------------------------------------------------------------------
_FLASH_AVAILABLE = False
_flash_attn_varlen_func = None
_flash_attn_func = None

try:
    from flash_attn import flash_attn_varlen_func as _fav, flash_attn_func as _fa
    _flash_attn_varlen_func = _fav
    _flash_attn_func = _fa
    _FLASH_AVAILABLE = True
    logger.debug("FlashAttention-2 imported successfully.")
except Exception as _fa_exc:  # ImportError, OSError, or ABI mismatch
    logger.debug("FlashAttention-2 not available: %s", _fa_exc)

# ---------------------------------------------------------------------------
# Optional import: PyTorch SDPA (available from PyTorch >= 2.0)
# ---------------------------------------------------------------------------
_SDPA_AVAILABLE = False
try:
    from torch.nn.functional import scaled_dot_product_attention as _sdpa
    _SDPA_AVAILABLE = True
    logger.debug("torch.nn.functional.scaled_dot_product_attention is available.")
except ImportError:
    logger.debug("torch.nn.functional.scaled_dot_product_attention not available.")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
# Set to the active path string at the first model build; read by get_attention_path().
ATTENTION_PATH: Optional[str] = None

# Flag: whether the active path has already been logged for the current model build.
_PATH_LOGGED: bool = False


def _detect_path() -> str:
    """
    Determine which attention path will be used based on availability.
    Order: flash -> sdpa -> eager.
    """
    if _FLASH_AVAILABLE:
        return "flash"
    if _SDPA_AVAILABLE:
        return "sdpa"
    return "eager"


def set_attention_path_for_new_build() -> None:
    """
    Called once at model construction.  Resolves and logs the active path,
    then resets the per-build logging flag so the path is logged exactly once
    per model build (on the first forward pass of that build).
    """
    global ATTENTION_PATH, _PATH_LOGGED
    ATTENTION_PATH = _detect_path()
    _PATH_LOGGED = False  # allow one log line on first forward of the new build


def get_attention_path() -> str:
    """Return the currently active attention path: 'flash', 'sdpa', or 'eager'."""
    if ATTENTION_PATH is None:
        return _detect_path()
    return ATTENTION_PATH


# ---------------------------------------------------------------------------
# Helper: build cu_seqlens for FlashAttention varlen API from attention_mask
# ---------------------------------------------------------------------------

def _mask_to_cu_seqlens(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Convert a standard HuggingFace-style attention mask (batch, seq_len) with
    values 0/1 into the (cu_seqlens, max_seqlen) format required by
    flash_attn_varlen_func.

    Returns
    -------
    cu_seqlens : torch.Tensor  shape (batch+1,)  dtype int32  on the same device
    max_seqlen : int           maximum sequence length in the batch (unpadded)
    """
    # attention_mask: 1 = real token, 0 = pad
    seqlens = attention_mask.sum(dim=1).to(torch.int32)          # (batch,)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0), (1, 0))     # (batch+1,)
    max_seqlen = int(seqlens.max().item())
    return cu_seqlens, max_seqlen


def _pack_sequences(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Remove padding and pack real tokens into a flat (total_tokens, hidden) tensor.
    Needed for the varlen FlashAttention API.

    Returns
    -------
    x_packed    : (total_tokens, hidden)
    cu_seqlens  : (batch+1,) int32
    max_seqlen  : int
    """
    batch, seq_len, hidden = hidden_states.shape
    mask_bool = attention_mask.bool()                              # (batch, seq_len)
    x_packed = hidden_states[mask_bool]                            # (total_tokens, hidden)
    cu_seqlens, max_seqlen = _mask_to_cu_seqlens(attention_mask)
    return x_packed, cu_seqlens, max_seqlen


def _unpack_sequences(
    x_packed: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Re-insert padding zeros to recover the (batch, seq_len, hidden) layout.
    """
    batch, seq_len = attention_mask.shape
    hidden = x_packed.shape[-1]
    out = torch.zeros(batch, seq_len, hidden, dtype=x_packed.dtype, device=x_packed.device)
    mask_bool = attention_mask.bool()
    out[mask_bool] = x_packed
    return out


# ---------------------------------------------------------------------------
# Core: BidirectionalSelfAttention
# ---------------------------------------------------------------------------

class BidirectionalSelfAttention(nn.Module):
    """
    Multi-head self-attention for bidirectional (MLM) transformers.

    Attention path selection (in priority order):
        1. FlashAttention-2 varlen  - padding-aware, BF16 only.
        2. PyTorch SDPA             - fused kernel where available.
        3. Eager                    - plain Q @ K^T / sqrt(d) + mask + softmax @ V.

    The selected path is logged once per model build on the first forward pass.

    Parameters
    ----------
    hidden_size : int
    num_attention_heads : int
    attention_probs_dropout_prob : float  (default 0.1)
    use_bf16 : bool  (default True)  -- cast inputs to BF16 for the flash path.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
        use_bf16: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})."
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_bf16 = use_bf16

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Dropout (used in eager and SDPA paths; FA2 has its own dropout arg)
        self.attn_dropout = nn.Dropout(p=attention_probs_dropout_prob)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, hidden) -> (batch, heads, seq, head_dim)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_attention_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, heads, seq, head_dim) -> (batch, seq, hidden)"""
        batch, heads, seq, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq, heads * head_dim)

    # ------------------------------------------------------------------
    # Attention implementations
    # ------------------------------------------------------------------

    def _flash_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        FlashAttention-2 path using the varlen (padding-aware) API.
        Inputs are cast to BF16 as required by FlashAttention.

        If attention_mask is None, fall back to the padded flash_attn_func.
        """
        batch, seq_len, hidden = hidden_states.shape
        dtype_orig = hidden_states.dtype

        # BF16 is required
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        dropout_p = self.attention_probs_dropout_prob if self.training else 0.0

        if attention_mask is not None:
            # Varlen path: pack sequences, call flash_attn_varlen_func, unpack
            x_packed, cu_seqlens, max_seqlen = _pack_sequences(hidden_states, attention_mask)
            q_p, cu_q, mq = _pack_sequences(q, attention_mask)
            k_p, cu_k, mk = _pack_sequences(k, attention_mask)
            v_p, cu_v, mv = _pack_sequences(v, attention_mask)

            # FlashAttention varlen expects (total, nheads, head_dim)
            q_p = q_p.view(-1, self.num_attention_heads, self.head_dim)
            k_p = k_p.view(-1, self.num_attention_heads, self.head_dim)
            v_p = v_p.view(-1, self.num_attention_heads, self.head_dim)

            out_packed = _flash_attn_varlen_func(
                q_p, k_p, v_p,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=mq,
                max_seqlen_k=mk,
                dropout_p=dropout_p,
                causal=False,           # bidirectional for MLM
            )                           # -> (total, nheads, head_dim)

            out_packed = out_packed.view(-1, hidden)
            out = _unpack_sequences(out_packed, attention_mask)
        else:
            # No mask: use the simpler padded flash_attn_func
            # Expects (batch, seqlen, nheads, head_dim)
            q4 = q.view(batch, seq_len, self.num_attention_heads, self.head_dim)
            k4 = k.view(batch, seq_len, self.num_attention_heads, self.head_dim)
            v4 = v.view(batch, seq_len, self.num_attention_heads, self.head_dim)
            out4 = _flash_attn_func(
                q4, k4, v4,
                dropout_p=dropout_p,
                causal=False,
            )                           # -> (batch, seqlen, nheads, head_dim)
            out = out4.view(batch, seq_len, hidden)

        out = out.to(dtype_orig)
        return self.out_proj(out)

    def _sdpa_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """PyTorch SDPA path (fused kernel when available)."""
        q = self._split_heads(self.q_proj(hidden_states))   # (b, h, s, d)
        k = self._split_heads(self.k_proj(hidden_states))
        v = self._split_heads(self.v_proj(hidden_states))

        # Build additive mask for SDPA: 0 for real tokens, -inf for padding
        bias = None
        if attention_mask is not None:
            # attention_mask: (batch, seq) with 1 = real, 0 = pad
            additive = (1.0 - attention_mask.float()) * torch.finfo(hidden_states.dtype).min
            # Broadcast to (batch, 1, 1, seq)
            bias = additive.unsqueeze(1).unsqueeze(2)

        dropout_p = self.attention_probs_dropout_prob if self.training else 0.0

        # scaled_dot_product_attention signature:
        # (query, key, value, attn_mask, dropout_p, is_causal)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,
            dropout_p=dropout_p,
            is_causal=False,
        )   # (batch, heads, seq, head_dim)

        out = self._merge_heads(out)    # (batch, seq, hidden)
        return self.out_proj(out)

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Eager (manual) attention path."""
        q = self._split_heads(self.q_proj(hidden_states))   # (b, h, s, d)
        k = self._split_heads(self.k_proj(hidden_states))
        v = self._split_heads(self.v_proj(hidden_states))

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (b, h, s, s)

        if attention_mask is not None:
            additive = (1.0 - attention_mask.float()) * torch.finfo(scores.dtype).min
            scores = scores + additive.unsqueeze(1).unsqueeze(2)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)                     # (b, h, s, d)
        out = self._merge_heads(out)                            # (b, s, hidden)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (batch, seq_len, hidden_size)
        attention_mask : (batch, seq_len) with 1 for real tokens, 0 for padding.
                         May be None if no padding is present.

        Returns
        -------
        output : (batch, seq_len, hidden_size)
        """
        global _PATH_LOGGED

        # Log the active path once per model build (first forward pass)
        if not _PATH_LOGGED:
            path = get_attention_path()
            logger.info(
                "Attention path active for this model build: %s  "
                "(flash_available=%s, sdpa_available=%s)",
                path.upper(), _FLASH_AVAILABLE, _SDPA_AVAILABLE,
            )
            _PATH_LOGGED = True

        path = get_attention_path()

        # FlashAttention path: require BF16-capable input, CUDA device, and no CPU
        if path == "flash":
            eligible = (
                hidden_states.is_cuda
                and (hidden_states.dtype in (torch.bfloat16, torch.float16, torch.float32))
            )
            if eligible:
                try:
                    return self._flash_forward(hidden_states, attention_mask)
                except Exception as exc:
                    logger.warning(
                        "FlashAttention-2 forward failed (%s); falling back to SDPA/eager.", exc
                    )
                    # Fall through to next path
            else:
                logger.debug(
                    "FlashAttention-2 not eligible for this input (device=%s, dtype=%s); "
                    "falling back.",
                    hidden_states.device, hidden_states.dtype,
                )

        # SDPA path
        if _SDPA_AVAILABLE:
            try:
                return self._sdpa_forward(hidden_states, attention_mask)
            except Exception as exc:
                logger.warning(
                    "SDPA forward failed (%s); falling back to eager attention.", exc
                )

        # Eager fallback (always safe)
        return self._eager_forward(hidden_states, attention_mask)


# ---------------------------------------------------------------------------
# BertLayer: attention + FFN (standard BERT post-norm style)
# ---------------------------------------------------------------------------

class BertLayer(nn.Module):
    """
    Standard BERT encoder layer.

    Architecture (post-norm, as in the original BERT paper):
        x -> Attention(x) + x -> LayerNorm -> FFN(x) + x -> LayerNorm

    Uses BidirectionalSelfAttention for the attention sub-layer.

    Parameters
    ----------
    hidden_size : int
    num_attention_heads : int
    intermediate_size : int  (FFN inner dimension)
    dropout : float          (applied after attention output and FFN output)
    attention_probs_dropout_prob : float
    use_bf16 : bool
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_bf16: bool = True,
    ) -> None:
        super().__init__()

        # Attention sub-layer
        self.attention = BidirectionalSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_bf16=use_bf16,
        )
        self.attn_dropout = nn.Dropout(p=dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # FFN sub-layer
        self.ffn_linear1 = nn.Linear(hidden_size, intermediate_size)
        self.ffn_activation = nn.GELU()
        self.ffn_linear2 = nn.Linear(intermediate_size, hidden_size)
        self.ffn_dropout = nn.Dropout(p=dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in (self.ffn_linear1, self.ffn_linear2):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (batch, seq_len, hidden_size)
        attention_mask : (batch, seq_len) or None

        Returns
        -------
        hidden_states : (batch, seq_len, hidden_size)
        """
        # --- Attention sub-layer (post-norm) ---
        attn_out = self.attention(hidden_states, attention_mask)
        attn_out = self.attn_dropout(attn_out)
        hidden_states = self.attn_layer_norm(hidden_states + attn_out)

        # --- FFN sub-layer (post-norm) ---
        ffn_out = self.ffn_linear2(self.ffn_activation(self.ffn_linear1(hidden_states)))
        ffn_out = self.ffn_dropout(ffn_out)
        hidden_states = self.ffn_layer_norm(hidden_states + ffn_out)

        return hidden_states


# Keep BertAttentionLayer as an alias to BertLayer for backward compatibility
# with any external code that imports it by the longer name.
BertAttentionLayer = BertLayer

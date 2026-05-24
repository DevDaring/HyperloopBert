"""
common/architectures.py

All five model architectures for the HyperloopBERT research pipeline.

All are encoder-only, bidirectional, GELU, absolute position embeddings,
MLM objective (15% masking: 80% [MASK] / 10% random / 10% original).
All are compute-matched at effective depth 12 (12 transformer-layer
applications per forward pass); they differ only in the number of
*unique* weight sets they hold.

Architecture summary:
  VanillaBERT          12 independent BertLayer instances
  LoopedBERT           begin(2) -> middle(2 x 4) -> end(2)  [unique=6]
  ALBERTLoopedBERT     1 shared BertLayer x 12              [unique=1]
  HyperloopBERT        LoopedBERT + num_streams=4 parallel streams + CWSA
  EarlyMergeHyperloop  HyperloopBERT with early stream-merge at merge_at
"""

# CITATION: Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional
#           Transformers for Language Understanding. NAACL 2019.
#           https://arxiv.org/abs/1810.04805   [VanillaBERT baseline]
# CITATION: Lan, Z. et al. (2020). ALBERT: A Lite BERT for Self-supervised
#           Learning of Language Representations. ICLR 2020.
#           https://arxiv.org/abs/1909.11942
#           Note: No embedding factorization here - isolates weight-sharing from
#           embedding compression to test the SCH directly.
# CITATION: Saunshi, N. et al. (2025). Reasoning with Latent Thoughts:
#           On the Power of Looped Transformers. arXiv.
#           [memorization-reasoning tradeoff = SCH basis]
# CITATION: Bae, J. et al. (2025). Looped encoder adaptation. [LoopedBERT]
# CITATION: Zeitoun, A., Torroba-Hennigen, L., & Kim, Y. (2026).
#           Hyperloop Transformers. arXiv:2604.21254. MIT.
#           [HyperloopBERT base; ours = first controlled encoder-only adaptation + CWSA]
# COUNTER:  Zhu, L. et al. (2025). arXiv:2603.08391.
#           [SCH counter-evidence: similar per-parameter memorization in looped models;
#           address head-on in paper, do not refute a priori]
# DATASET WARNING: Models trained here may encode stereotypical content by design
#           for fairness research. Research/fairness-audit use only.

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils

from common.attention import BertLayer, set_attention_path_for_new_build

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and size presets
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 30522
MAX_POSITION_EMBEDDINGS: int = 512
PAD_TOKEN_ID: int = 0
MASK_TOKEN_ID: int = 4       # [MASK] is index 4 in the standard BERT vocab
CLS_TOKEN_ID: int = 101
SEP_TOKEN_ID: int = 102

SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    "tiny": {
        "hidden_size": 256,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
    },
    "small": {
        "hidden_size": 512,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
    },
    "base": {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
}

# MLM masking probabilities (Devlin et al. 2019, Section 3.1)
_MLM_PROB: float = 0.15
_MASK_REPLACE_PROB: float = 0.80   # replace with [MASK]
_RANDOM_REPLACE_PROB: float = 0.10  # replace with random token
# remaining 0.10: keep original


# ---------------------------------------------------------------------------
# Shared embedding module
# ---------------------------------------------------------------------------

class BertEmbeddings(nn.Module):
    """
    Standard BERT input embeddings.

    Combines:
        word_embeddings       (vocab_size, hidden_size)
        position_embeddings   (MAX_POSITION_EMBEDDINGS, hidden_size)
        token_type_embeddings (2, hidden_size)

    Followed by LayerNorm + Dropout.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = MAX_POSITION_EMBEDDINGS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_TOKEN_ID)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)

        # Position ids buffer: persistent, not a parameter
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).unsqueeze(0),  # (1, max_pos)
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.token_type_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids       : (batch, seq_len)
        token_type_ids  : (batch, seq_len) or None -> defaults to zeros
        position_ids    : (batch, seq_len) or None -> defaults to 0..seq_len-1

        Returns
        -------
        embeddings : (batch, seq_len, hidden_size)
        """
        seq_len = input_ids.size(1)

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        tok_type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + pos_emb + tok_type_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# ---------------------------------------------------------------------------
# MLM head
# ---------------------------------------------------------------------------

class BertMLMHead(nn.Module):
    """
    MLM prediction head.

    Architecture:
        Linear(hidden, hidden) -> GELU -> LayerNorm -> Linear(hidden, vocab_size)

    The final projection weight is tied to the word embedding matrix after
    construction via tie_weights().
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)

        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def tie_weights(self, word_embedding_weight: nn.Parameter) -> None:
        """Tie decoder weight to the word embedding weight matrix."""
        self.decoder.weight = word_embedding_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (batch, seq_len, hidden_size)

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        return self.decoder(x)


# ---------------------------------------------------------------------------
# Pooler
# ---------------------------------------------------------------------------

class BertPooler(nn.Module):
    """
    Extract the [CLS] token representation and project it.
    Standard BERT pooler: Linear(hidden, hidden) -> Tanh.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (batch, seq_len, hidden_size)

        Returns
        -------
        pooled : (batch, hidden_size)
        """
        cls_token = hidden_states[:, 0, :]  # [CLS] is always position 0
        return self.activation(self.dense(cls_token))


# ---------------------------------------------------------------------------
# MLM masking utility
# ---------------------------------------------------------------------------

def apply_mlm_mask(
    input_ids: torch.Tensor,
    tokenizer_or_attention_mask,
    vocab_size: Optional[int] = None,
    *,
    prob: float = _MLM_PROB,
    mask_prob: float = _MASK_REPLACE_PROB,
    random_prob: float = _RANDOM_REPLACE_PROB,
    keep_prob: float = 0.10,
    mlm_probability: Optional[float] = None,
    mask_token_id: int = MASK_TOKEN_ID,
    pad_token_id: int = PAD_TOKEN_ID,
    cls_token_id: int = CLS_TOKEN_ID,
    sep_token_id: int = SEP_TOKEN_ID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply BERT-style MLM masking.

    Accepts two calling conventions:
      1. apply_mlm_mask(input_ids, tokenizer, prob=0.15, mask_prob=0.80, ...)
         (used by train scripts; derives vocab_size and special token IDs from tokenizer)
      2. apply_mlm_mask(input_ids, attention_mask_tensor, vocab_size, mlm_probability=0.15, ...)
         (legacy positional interface)

    15% of real (non-special) tokens are selected:
        80% replaced with [MASK]
        10% replaced with a random token
        10% kept as-is

    Returns
    -------
    masked_input_ids : (batch, seq_len) -- the corrupted input
    labels           : (batch, seq_len) -- original ids at masked positions, -100 elsewhere
    """
    # Detect which calling convention is used
    attention_mask: Optional[torch.Tensor]
    if isinstance(tokenizer_or_attention_mask, torch.Tensor):
        # Legacy: (input_ids, attention_mask, vocab_size, mlm_probability=...)
        attention_mask = tokenizer_or_attention_mask
        if vocab_size is None:
            raise ValueError("vocab_size must be provided when passing attention_mask as second arg.")
        mlm_prob = mlm_probability if mlm_probability is not None else prob
        mask_replace_prob = mask_prob
        random_replace_prob = random_prob
        v_size = vocab_size
        _mask_token_id = mask_token_id
        _pad_token_id = pad_token_id
        _cls_token_id = cls_token_id
        _sep_token_id = sep_token_id
    else:
        # Tokenizer-based: (input_ids, tokenizer, prob=..., mask_prob=..., random_prob=..., keep_prob=...)
        tokenizer = tokenizer_or_attention_mask
        attention_mask = None
        mlm_prob = prob
        mask_replace_prob = mask_prob
        random_replace_prob = random_prob
        v_size = tokenizer.vocab_size
        _mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else MASK_TOKEN_ID
        _pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else PAD_TOKEN_ID
        _cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else CLS_TOKEN_ID
        _sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else SEP_TOKEN_ID

    labels = input_ids.clone()

    # Build a probability matrix; zero out special tokens and padding
    prob_matrix = torch.full(input_ids.shape, mlm_prob, device=input_ids.device)
    special_tokens_mask = (
        (input_ids == _pad_token_id)
        | (input_ids == _cls_token_id)
        | (input_ids == _sep_token_id)
    )
    if attention_mask is not None:
        special_tokens_mask = special_tokens_mask | (~attention_mask.bool())

    prob_matrix.masked_fill_(special_tokens_mask, 0.0)

    # Which positions are masked?
    masked_positions = torch.bernoulli(prob_matrix).bool()
    labels[~masked_positions] = -100  # only compute loss on masked positions

    # mask_replace_prob -> [MASK]
    replace_with_mask = torch.bernoulli(
        torch.full(input_ids.shape, mask_replace_prob, device=input_ids.device)
    ).bool() & masked_positions
    input_ids = input_ids.clone()
    input_ids[replace_with_mask] = _mask_token_id

    # random_replace_prob / (1 - mask_replace_prob) of the remaining -> random token
    remaining = masked_positions & ~replace_with_mask
    remaining_random_prob = random_replace_prob / (1.0 - mask_replace_prob) if (1.0 - mask_replace_prob) > 0 else 0.5
    replace_with_random = torch.bernoulli(
        torch.full(input_ids.shape, remaining_random_prob, device=input_ids.device)
    ).bool() & remaining
    random_tokens = torch.randint(
        low=5,  # skip special tokens at indices 0-4
        high=v_size,
        size=input_ids.shape,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    input_ids[replace_with_random] = random_tokens[replace_with_random]

    # Remaining keep_prob -> unchanged (already in input_ids)

    return input_ids, labels


# ---------------------------------------------------------------------------
# Model info utility
# ---------------------------------------------------------------------------

def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Compute and return a dict with key model statistics.

    Returns
    -------
    dict with keys:
        Unique_Parameters  : int   -- parameters in unique (non-shared) weights
        Total_Parameters   : int   -- total parameters counting shared ones once
        Effective_Depth    : int   -- number of layer applications per forward pass
        Hidden_Size        : int
        Shared_Ratio       : float -- 1 - unique_layers / 12
    """
    # Total unique parameters (count each parameter tensor once)
    unique_params = sum(p.numel() for p in model.parameters())

    # Total parameters counting all forward-pass applications
    # (including shared ones counted once per sharing group = unique_params)
    # For reporting consistency, Total_Parameters here equals unique_params
    # because that is what fits on disk; Effective_Depth captures the reuse.
    effective_depth = getattr(model, "effective_depth", 12)
    hidden_size = getattr(model, "hidden_size", -1)
    shared_ratio = getattr(model, "shared_ratio", 0.0)

    return {
        "Unique_Parameters": unique_params,
        "Total_Parameters": unique_params,  # disk footprint = unique params
        "Effective_Depth": effective_depth,
        "Hidden_Size": hidden_size,
        "Shared_Ratio": shared_ratio,
    }


# ---------------------------------------------------------------------------
# Base class with common gradient-checkpointing support
# ---------------------------------------------------------------------------

class _BaseModel(nn.Module):
    """
    Base class providing:
    - gradient_checkpointing_enable() / gradient_checkpointing_disable()
    - a _maybe_checkpoint() wrapper for forward passes
    """

    def __init__(self) -> None:
        super().__init__()
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing to trade compute for memory."""
        self._gradient_checkpointing = True
        logger.debug("%s: gradient checkpointing enabled.", self.__class__.__name__)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def _checkpoint_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply a single BertLayer, optionally under gradient checkpointing.
        torch.utils.checkpoint.checkpoint does not handle None arguments
        gracefully in all versions, so we use a wrapper closure.
        """
        if self._gradient_checkpointing and self.training:
            def _forward(hs: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
                return layer(hs, mask)

            if attention_mask is None:
                # checkpoint requires at least one tensor; provide a dummy
                dummy = torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype)
                def _fwd_no_mask(hs: torch.Tensor, _dummy: torch.Tensor) -> torch.Tensor:
                    return layer(hs, None)
                return checkpoint_utils.checkpoint(_fwd_no_mask, hidden_states, dummy, use_reentrant=False)
            else:
                return checkpoint_utils.checkpoint(_forward, hidden_states, attention_mask, use_reentrant=False)
        else:
            return layer(hidden_states, attention_mask)


# ---------------------------------------------------------------------------
# VanillaBERT
# ---------------------------------------------------------------------------

# CITATION: Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional
#           Transformers for Language Understanding. NAACL 2019.
#           [VanillaBERT baseline -- 12 independent layers, effective_depth=12]

class VanillaBERT(_BaseModel):
    """
    Standard BERT encoder.

    12 independent BertLayer instances applied sequentially.
    Effective depth = Unique layers = 12.  Shared_Ratio = 0.0.

    Returns
    -------
    dict with keys:
        last_hidden_state : (batch, seq_len, hidden_size)
        pooler_output     : (batch, hidden_size)
        mlm_logits        : (batch, seq_len, vocab_size)
    """

    effective_depth: int = 12
    unique_layers: int = 12
    shared_ratio: float = 0.0

    def __init__(
        self,
        size: str = "base",
        vocab_size: int = VOCAB_SIZE,
        max_position_embeddings: int = MAX_POSITION_EMBEDDINGS,
        dropout: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_bf16: bool = True,
    ) -> None:
        super().__init__()
        if size not in SIZE_PRESETS:
            raise ValueError(f"Unknown size preset '{size}'. Choose from {list(SIZE_PRESETS)}.")
        preset = SIZE_PRESETS[size]
        self.hidden_size = preset["hidden_size"]
        num_heads = preset["num_attention_heads"]
        intermediate = preset["intermediate_size"]
        self.vocab_size = vocab_size

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )

        self.encoder = nn.ModuleList([
            BertLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=num_heads,
                intermediate_size=intermediate,
                dropout=dropout,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                use_bf16=use_bf16,
            )
            for _ in range(12)
        ])

        self.pooler = BertPooler(self.hidden_size)
        self.mlm_head = BertMLMHead(self.hidden_size, vocab_size)
        self.mlm_head.tie_weights(self.embeddings.word_embeddings.weight)

        set_attention_path_for_new_build()
        info = get_model_info(self)
        logger.info(
            "VanillaBERT (%s) constructed: hidden=%d, heads=%d, intermediate=%d, "
            "unique_params=%d, effective_depth=%d, shared_ratio=%.4f",
            size, self.hidden_size, num_heads, intermediate,
            info["Unique_Parameters"], self.effective_depth, self.shared_ratio,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids)

        for layer in self.encoder:
            hidden_states = self._checkpoint_layer(layer, hidden_states, attention_mask)

        pooler_output = self.pooler(hidden_states)
        mlm_logits = self.mlm_head(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooler_output,
            "mlm_logits": mlm_logits,
        }


# ---------------------------------------------------------------------------
# LoopedBERT
# ---------------------------------------------------------------------------

# CITATION: Saunshi, N. et al. (2025). Reasoning with Latent Thoughts:
#           On the Power of Looped Transformers. arXiv.
#           [memorization-reasoning tradeoff is the SCH theoretical basis]
# CITATION: Bae, J. et al. (2025). Looped encoder adaptation. [LoopedBERT design]
# COUNTER:  Zhu, L. et al. (2025). arXiv:2603.08391.
#           [counter-evidence: similar per-parameter memorization in looped models;
#           addressed head-on in the paper, not refuted a priori]

class LoopedBERT(_BaseModel):
    """
    Looped BERT encoder.

    Architecture:
        begin   : 2 independent BertLayers
        middle  : 2 shared BertLayers  (ONE set, applied 4 times in a loop)
        end     : 2 independent BertLayers

    Total unique layers = 6.  Effective depth = 2 + 2*4 + 2 = 12.
    Shared_Ratio = 1 - 6/12 = 0.5.

    Loop-index embeddings (nn.Embedding(4, hidden_size)) are added to the
    hidden state BEFORE each middle-block iteration to break symmetry across
    loop iterations.

    Returns
    -------
    dict with keys:
        last_hidden_state : (batch, seq_len, hidden_size)
        pooler_output     : (batch, hidden_size)
        mlm_logits        : (batch, seq_len, vocab_size)
    """

    effective_depth: int = 12
    unique_layers: int = 6
    shared_ratio: float = 0.5
    num_middle_loops: int = 4

    def __init__(
        self,
        size: str = "base",
        vocab_size: int = VOCAB_SIZE,
        max_position_embeddings: int = MAX_POSITION_EMBEDDINGS,
        dropout: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_bf16: bool = True,
    ) -> None:
        super().__init__()
        if size not in SIZE_PRESETS:
            raise ValueError(f"Unknown size preset '{size}'. Choose from {list(SIZE_PRESETS)}.")
        preset = SIZE_PRESETS[size]
        self.hidden_size = preset["hidden_size"]
        num_heads = preset["num_attention_heads"]
        intermediate = preset["intermediate_size"]
        self.vocab_size = vocab_size

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )

        layer_kwargs = dict(
            hidden_size=self.hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=intermediate,
            dropout=dropout,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_bf16=use_bf16,
        )

        # begin: 2 independent layers
        self.begin_layers = nn.ModuleList([BertLayer(**layer_kwargs) for _ in range(2)])

        # middle: 2 shared layers (ONE set)
        self.middle_layers = nn.ModuleList([BertLayer(**layer_kwargs) for _ in range(2)])

        # end: 2 independent layers
        self.end_layers = nn.ModuleList([BertLayer(**layer_kwargs) for _ in range(2)])

        # Loop-index embeddings: one per loop iteration (4 iterations)
        self.loop_index_embeddings = nn.Embedding(self.num_middle_loops, self.hidden_size)
        nn.init.normal_(self.loop_index_embeddings.weight, mean=0.0, std=0.02)

        self.pooler = BertPooler(self.hidden_size)
        self.mlm_head = BertMLMHead(self.hidden_size, vocab_size)
        self.mlm_head.tie_weights(self.embeddings.word_embeddings.weight)

        set_attention_path_for_new_build()
        info = get_model_info(self)
        logger.info(
            "LoopedBERT (%s) constructed: hidden=%d, heads=%d, intermediate=%d, "
            "unique_params=%d, effective_depth=%d, unique_layers=%d, shared_ratio=%.4f",
            size, self.hidden_size, num_heads, intermediate,
            info["Unique_Parameters"], self.effective_depth,
            self.unique_layers, self.shared_ratio,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids)

        # Begin block (2 layers)
        for layer in self.begin_layers:
            hidden_states = self._checkpoint_layer(layer, hidden_states, attention_mask)

        # Middle block: shared 2-layer block looped 4 times
        for loop_idx in range(self.num_middle_loops):
            # Add loop-index embedding to break symmetry across iterations
            loop_emb = self.loop_index_embeddings(
                torch.tensor(loop_idx, device=hidden_states.device, dtype=torch.long)
            )  # (hidden_size,)
            hidden_states = hidden_states + loop_emb.unsqueeze(0).unsqueeze(0)

            for layer in self.middle_layers:
                hidden_states = self._checkpoint_layer(layer, hidden_states, attention_mask)

        # End block (2 layers)
        for layer in self.end_layers:
            hidden_states = self._checkpoint_layer(layer, hidden_states, attention_mask)

        pooler_output = self.pooler(hidden_states)
        mlm_logits = self.mlm_head(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooler_output,
            "mlm_logits": mlm_logits,
        }


# ---------------------------------------------------------------------------
# ALBERTLoopedBERT
# ---------------------------------------------------------------------------

# CITATION: Lan, Z. et al. (2020). ALBERT: A Lite BERT for Self-supervised
#           Learning of Language Representations. ICLR 2020.
#           https://arxiv.org/abs/1909.11942
#           Note: No embedding factorization -- isolates weight-sharing from
#           embedding compression. Standard hidden_size embeddings are used.
# CITATION: Saunshi, N. et al. (2025). Reasoning with Latent Thoughts:
#           On the Power of Looped Transformers. arXiv.  [SCH basis]

class ALBERTLoopedBERT(_BaseModel):
    """
    ALBERT-style extreme parameter sharing: ONE BertLayer applied 12 times.

    Unique layers = 1.  Effective depth = 12.
    Shared_Ratio = 1 - 1/12 = 0.9167.

    No embedding factorization (isolates weight-sharing from embedding
    compression to directly test the SCH).

    Loop-index embeddings (nn.Embedding(12, hidden_size)) are added before
    each of the 12 applications to break representational symmetry.

    Returns
    -------
    dict with keys:
        last_hidden_state : (batch, seq_len, hidden_size)
        pooler_output     : (batch, hidden_size)
        mlm_logits        : (batch, seq_len, vocab_size)
    """

    effective_depth: int = 12
    unique_layers: int = 1
    shared_ratio: float = 1.0 - 1.0 / 12.0  # ~0.9167
    num_loops: int = 12

    def __init__(
        self,
        size: str = "base",
        vocab_size: int = VOCAB_SIZE,
        max_position_embeddings: int = MAX_POSITION_EMBEDDINGS,
        dropout: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_bf16: bool = True,
    ) -> None:
        super().__init__()
        if size not in SIZE_PRESETS:
            raise ValueError(f"Unknown size preset '{size}'. Choose from {list(SIZE_PRESETS)}.")
        preset = SIZE_PRESETS[size]
        self.hidden_size = preset["hidden_size"]
        num_heads = preset["num_attention_heads"]
        intermediate = preset["intermediate_size"]
        self.vocab_size = vocab_size

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )

        # ONE shared layer
        self.shared_layer = BertLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=intermediate,
            dropout=dropout,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_bf16=use_bf16,
        )

        # Loop-index embeddings: one per loop (12 iterations)
        self.loop_index_embeddings = nn.Embedding(self.num_loops, self.hidden_size)
        nn.init.normal_(self.loop_index_embeddings.weight, mean=0.0, std=0.02)

        self.pooler = BertPooler(self.hidden_size)
        self.mlm_head = BertMLMHead(self.hidden_size, vocab_size)
        self.mlm_head.tie_weights(self.embeddings.word_embeddings.weight)

        set_attention_path_for_new_build()
        info = get_model_info(self)
        logger.info(
            "ALBERTLoopedBERT (%s) constructed: hidden=%d, heads=%d, intermediate=%d, "
            "unique_params=%d, effective_depth=%d, unique_layers=%d, shared_ratio=%.4f",
            size, self.hidden_size, num_heads, intermediate,
            info["Unique_Parameters"], self.effective_depth,
            self.unique_layers, self.shared_ratio,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids)

        for loop_idx in range(self.num_loops):
            loop_emb = self.loop_index_embeddings(
                torch.tensor(loop_idx, device=hidden_states.device, dtype=torch.long)
            )  # (hidden_size,)
            hidden_states = hidden_states + loop_emb.unsqueeze(0).unsqueeze(0)
            hidden_states = self._checkpoint_layer(self.shared_layer, hidden_states, attention_mask)

        pooler_output = self.pooler(hidden_states)
        mlm_logits = self.mlm_head(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooler_output,
            "mlm_logits": mlm_logits,
        }


# ---------------------------------------------------------------------------
# HyperloopBERT  (Stage 3)
# ---------------------------------------------------------------------------

# CITATION: Zeitoun, A., Torroba-Hennigen, L., & Kim, Y. (2026).
#           Hyperloop Transformers. arXiv:2604.21254. MIT.
#           [original Hyperloop design; ours = first controlled encoder-only
#           adaptation + CLS-Weighted Stream Aggregation (CWSA)]
# CITATION: Saunshi, N. et al. (2025). Reasoning with Latent Thoughts:
#           On the Power of Looped Transformers. arXiv.  [SCH / loop basis]
# CITATION: Bae, J. et al. (2025). Looped encoder adaptation.  [LoopedBERT basis]
# COUNTER:  Zhu, L. et al. (2025). arXiv:2603.08391.
#           [SCH counter-evidence; addressed head-on in the paper]

class HyperloopBERT(_BaseModel):
    """
    HyperloopBERT encoder with CWSA (CLS-Weighted Stream Aggregation).

    Architecture:
        begin  : 2 independent BertLayers  (single stream)
        middle : 2 shared BertLayers, looped 4 times, with num_streams=4
                 parallel residual streams
        end    : 2 independent BertLayers  (after CWSA stream aggregation)

    Unique layers = 6.  Effective depth = 2 + 2*4 + 2 = 12.
    Shared_Ratio = 0.5.

    Stream processing (per loop iteration):
        1. depth_proj  : linear(num_streams * hidden_size -> hidden_size)
                         Mixes all streams into a single block input.
        2. middle block: applied ONCE on the mixed input (not once per stream).
                         This ensures compute-matching with LoopedBERT.
        3. width_proj  : linear(hidden_size -> num_streams * hidden_size)
                         Scatters the output back into all streams residually.
                         new_stream_i = old_stream_i + scattered_output_i

    CWSA (CLS-Weighted Stream Aggregation):
        After the final loop, compute soft-attention weights over each stream's
        [CLS] representation.  The final hidden state is the weighted sum of
        streams, using these CLS-derived weights.

        For each stream s:  score_s = sigmoid(cls_weight_linear(stream_s[:, 0, :]))
        Weights are softmax-normalised across streams.
        Final hidden state: sum_s (weight_s * stream_s)

    Stream snapshots (for mechanistic analysis):
        stream_snapshots: dict mapping loop_idx -> list of stream tensors
        Each entry is a list of num_streams tensors of shape (batch, seq_len, hidden_size).
        Stored as detached CPU tensors in eval mode; as live tensors during training
        (only the dict structure is preserved; actual storage enabled by
        model.enable_stream_snapshots = True to avoid memory overhead during training).

    Returns
    -------
    dict with keys:
        last_hidden_state  : (batch, seq_len, hidden_size)
        pooler_output      : (batch, hidden_size)
        mlm_logits         : (batch, seq_len, vocab_size)
        stream_snapshots   : dict[int, list[Tensor]]
    """

    effective_depth: int = 12
    unique_layers: int = 6
    shared_ratio: float = 0.5
    num_middle_loops: int = 4

    def __init__(
        self,
        size: str = "base",
        vocab_size: int = VOCAB_SIZE,
        max_position_embeddings: int = MAX_POSITION_EMBEDDINGS,
        num_streams: int = 4,
        dropout: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_bf16: bool = True,
    ) -> None:
        super().__init__()
        if size not in SIZE_PRESETS:
            raise ValueError(f"Unknown size preset '{size}'. Choose from {list(SIZE_PRESETS)}.")
        preset = SIZE_PRESETS[size]
        self.hidden_size = preset["hidden_size"]
        num_heads = preset["num_attention_heads"]
        intermediate = preset["intermediate_size"]
        self.vocab_size = vocab_size
        self.num_streams = num_streams
        # Enable snapshot collection (set to True before mechanistic analysis)
        self.enable_stream_snapshots: bool = False

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )

        layer_kwargs = dict(
            hidden_size=self.hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=intermediate,
            dropout=dropout,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_bf16=use_bf16,
        )

        # begin: 2 independent layers
        self.begin_layers = nn.ModuleList([BertLayer(**layer_kwargs) for _ in range(2)])

        # middle: 2 shared layers (ONE set, compute-matched to LoopedBERT)
        self.middle_layers = nn.ModuleList([BertLayer(**layer_kwargs) for _ in range(2)])

        # end: 2 independent layers
        self.end_layers = nn.ModuleList([BertLayer(**layer_kwargs) for _ in range(2)])

        # Hyper-connection projections: one depth_proj + one width_proj per loop
        self.depth_projs = nn.ModuleList([
            nn.Linear(num_streams * self.hidden_size, self.hidden_size, bias=True)
            for _ in range(self.num_middle_loops)
        ])
        self.width_projs = nn.ModuleList([
            nn.Linear(self.hidden_size, num_streams * self.hidden_size, bias=True)
            for _ in range(self.num_middle_loops)
        ])

        # Loop-index embeddings: one per loop iteration
        self.loop_index_embeddings = nn.Embedding(self.num_middle_loops, self.hidden_size)
        nn.init.normal_(self.loop_index_embeddings.weight, mean=0.0, std=0.02)

        # CWSA: linear to compute per-stream scalar score from [CLS] repr
        self.cwsa_linear = nn.Linear(self.hidden_size, 1, bias=True)

        # Initialize hyper-connection projections carefully
        self._init_hyperconnection_weights()

        self.pooler = BertPooler(self.hidden_size)
        self.mlm_head = BertMLMHead(self.hidden_size, vocab_size)
        self.mlm_head.tie_weights(self.embeddings.word_embeddings.weight)

        set_attention_path_for_new_build()
        info = get_model_info(self)
        logger.info(
            "HyperloopBERT (%s) constructed: hidden=%d, heads=%d, intermediate=%d, "
            "num_streams=%d, unique_params=%d, effective_depth=%d, "
            "unique_layers=%d, shared_ratio=%.4f",
            size, self.hidden_size, num_heads, intermediate,
            self.num_streams, info["Unique_Parameters"],
            self.effective_depth, self.unique_layers, self.shared_ratio,
        )

    def _init_hyperconnection_weights(self) -> None:
        """
        Initialise depth/width projections so that at construction, the
        model behaves approximately like LoopedBERT (identity flow through
        streams).  Specifically: depth_proj is initialised to average the
        streams (1/num_streams), width_proj to replicate output to all streams.
        """
        for depth_proj in self.depth_projs:
            nn.init.normal_(depth_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(depth_proj.bias)
        for width_proj in self.width_projs:
            nn.init.normal_(width_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(width_proj.bias)
        nn.init.normal_(self.cwsa_linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cwsa_linear.bias)

    def _apply_middle_block(
        self,
        mixed_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the shared 2-layer middle block to a single (mixed) input."""
        h = mixed_input
        for layer in self.middle_layers:
            h = self._checkpoint_layer(layer, h, attention_mask)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        input_ids       : (batch, seq_len)
        attention_mask  : (batch, seq_len) or None
        token_type_ids  : (batch, seq_len) or None
        position_ids    : (batch, seq_len) or None

        Returns
        -------
        dict with keys: last_hidden_state, pooler_output, mlm_logits, stream_snapshots
        """
        stream_snapshots: Dict[int, List[torch.Tensor]] = {}

        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids)

        # Begin block (single stream, 2 layers)
        for layer in self.begin_layers:
            hidden_states = self._checkpoint_layer(layer, hidden_states, attention_mask)

        # Initialise num_streams streams as copies of the begin output
        # streams: list of num_streams tensors, each (batch, seq_len, hidden_size)
        streams: List[torch.Tensor] = [hidden_states.clone() for _ in range(self.num_streams)]

        # Middle looped block (4 iterations of the shared 2-layer block)
        for loop_idx in range(self.num_middle_loops):
            # Optionally capture stream snapshots for mechanistic analysis
            if self.enable_stream_snapshots:
                if not self.training:
                    stream_snapshots[loop_idx] = [s.detach().cpu() for s in streams]
                else:
                    stream_snapshots[loop_idx] = [s.detach() for s in streams]

            # Loop-index embedding (added to mixed input to break symmetry)
            loop_emb = self.loop_index_embeddings(
                torch.tensor(loop_idx, device=hidden_states.device, dtype=torch.long)
            )  # (hidden_size,)

            # depth_connection: mix all streams into one block input
            # Concatenate streams along hidden dim: (batch, seq_len, num_streams * hidden)
            concatenated = torch.cat(streams, dim=-1)
            mixed = self.depth_projs[loop_idx](concatenated)   # (batch, seq_len, hidden)
            mixed = mixed + loop_emb.unsqueeze(0).unsqueeze(0)

            # Apply the shared middle block ONCE (compute-matched to LoopedBERT)
            block_output = self._apply_middle_block(mixed, attention_mask)
            # block_output: (batch, seq_len, hidden_size)

            # width_connection: scatter output back to all streams residually
            scattered = self.width_projs[loop_idx](block_output)
            # scattered: (batch, seq_len, num_streams * hidden)
            scattered_per_stream = scattered.chunk(self.num_streams, dim=-1)
            # Each chunk: (batch, seq_len, hidden_size)

            streams = [
                streams[i] + scattered_per_stream[i]
                for i in range(self.num_streams)
            ]

        # Capture final stream snapshots (after last loop)
        if self.enable_stream_snapshots:
            final_key = self.num_middle_loops
            if not self.training:
                stream_snapshots[final_key] = [s.detach().cpu() for s in streams]
            else:
                stream_snapshots[final_key] = [s.detach() for s in streams]

        # CWSA: CLS-Weighted Stream Aggregation
        # For each stream, compute a scalar score from the [CLS] token (position 0)
        cls_reps = torch.stack(
            [s[:, 0, :] for s in streams], dim=1
        )  # (batch, num_streams, hidden_size)

        # Score each stream via a shared linear + softmax
        scores = self.cwsa_linear(cls_reps).squeeze(-1)    # (batch, num_streams)
        weights = F.softmax(scores, dim=-1)                 # (batch, num_streams)

        # Weighted sum over streams for the full sequence
        # weights: (batch, num_streams) -> (batch, num_streams, 1, 1) for broadcasting
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (batch, num_streams, 1, 1)
        stacked_streams = torch.stack(streams, dim=1)            # (batch, num_streams, seq, hidden)
        aggregated = (weights_expanded * stacked_streams).sum(dim=1)  # (batch, seq, hidden)

        # End block (2 layers, single aggregated stream)
        for layer in self.end_layers:
            aggregated = self._checkpoint_layer(layer, aggregated, attention_mask)

        pooler_output = self.pooler(aggregated)
        mlm_logits = self.mlm_head(aggregated)

        return {
            "last_hidden_state": aggregated,
            "pooler_output": pooler_output,
            "mlm_logits": mlm_logits,
            "stream_snapshots": stream_snapshots,
        }


# ---------------------------------------------------------------------------
# EarlyMergeHyperloopBERT  (Stage 3 -- OOD intervention)
# ---------------------------------------------------------------------------

# CITATION: Zeitoun, A., Torroba-Hennigen, L., & Kim, Y. (2026).
#           Hyperloop Transformers. arXiv:2604.21254. MIT.
#           [base design; early-merge = our ablation, not from paper]
# NOTE:     EarlyMerge is an out-of-distribution (OOD) intervention, not a
#           causally clean experimental condition.  Do NOT label it as causal
#           proof in the paper.  Label explicitly as "OOD ablation".

class EarlyMergeHyperloopBERT(_BaseModel):
    """
    HyperloopBERT with early stream merge (OOD intervention for Stage 3).

    At loop iteration `merge_at` (1-indexed, in {1, 2, 3}), all num_streams
    streams are merged into a single stream by CWSA, and subsequent iterations
    proceed as a single-stream LoopedBERT (no further width/depth projections).

    Labeled "OOD ablation" in all result files.  Not a causal proof.

    Architecture:
        Loop iterations 1 .. merge_at      : multi-stream HyperloopBERT
        Loop iterations merge_at+1 .. 4   : single stream (like LoopedBERT)

    Parameters
    ----------
    merge_at : int  in {1, 2, 3}
        Loop iteration index (1-indexed) after which streams are merged.
    """

    effective_depth: int = 12
    unique_layers: int = 6
    shared_ratio: float = 0.5
    num_middle_loops: int = 4

    def __init__(
        self,
        size: str = "base",
        vocab_size: int = VOCAB_SIZE,
        max_position_embeddings: int = MAX_POSITION_EMBEDDINGS,
        num_streams: int = 4,
        merge_at: int = 2,
        dropout: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_bf16: bool = True,
    ) -> None:
        super().__init__()
        if size not in SIZE_PRESETS:
            raise ValueError(f"Unknown size preset '{size}'. Choose from {list(SIZE_PRESETS)}.")
        if merge_at not in {1, 2, 3}:
            raise ValueError(f"merge_at must be in {{1, 2, 3}}, got {merge_at}.")
        preset = SIZE_PRESETS[size]
        self.hidden_size = preset["hidden_size"]
        num_heads = preset["num_attention_heads"]
        intermediate = preset["intermediate_size"]
        self.vocab_size = vocab_size
        self.num_streams = num_streams
        self.merge_at = merge_at
        self.enable_stream_snapshots: bool = False

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )

        layer_kwargs = dict(
            hidden_size=self.hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=intermediate,
            dropout=dropout,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_bf16=use_bf16,
        )

        self.begin_layers = nn.ModuleList([BertLayer(**layer_kwargs) for _ in range(2)])
        self.middle_layers = nn.ModuleList([BertLayer(**layer_kwargs) for _ in range(2)])
        self.end_layers = nn.ModuleList([BertLayer(**layer_kwargs) for _ in range(2)])

        # Hyper-connection projections (only used for iterations 1..merge_at)
        self.depth_projs = nn.ModuleList([
            nn.Linear(num_streams * self.hidden_size, self.hidden_size, bias=True)
            for _ in range(self.num_middle_loops)
        ])
        self.width_projs = nn.ModuleList([
            nn.Linear(self.hidden_size, num_streams * self.hidden_size, bias=True)
            for _ in range(self.num_middle_loops)
        ])

        self.loop_index_embeddings = nn.Embedding(self.num_middle_loops, self.hidden_size)
        nn.init.normal_(self.loop_index_embeddings.weight, mean=0.0, std=0.02)

        # CWSA for the merge point (and end)
        self.cwsa_linear = nn.Linear(self.hidden_size, 1, bias=True)

        # Initialise projections
        for proj in list(self.depth_projs) + list(self.width_projs):
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(proj.bias)
        nn.init.normal_(self.cwsa_linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cwsa_linear.bias)

        self.pooler = BertPooler(self.hidden_size)
        self.mlm_head = BertMLMHead(self.hidden_size, vocab_size)
        self.mlm_head.tie_weights(self.embeddings.word_embeddings.weight)

        set_attention_path_for_new_build()
        info = get_model_info(self)
        logger.info(
            "EarlyMergeHyperloopBERT (%s, merge_at=%d) constructed: "
            "hidden=%d, heads=%d, intermediate=%d, num_streams=%d, "
            "unique_params=%d, effective_depth=%d, shared_ratio=%.4f",
            size, self.merge_at, self.hidden_size, num_heads, intermediate,
            self.num_streams, info["Unique_Parameters"],
            self.effective_depth, self.shared_ratio,
        )

    def _cwsa_merge(self, streams: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge a list of streams into one via CWSA.

        Parameters
        ----------
        streams : list of (batch, seq_len, hidden_size)

        Returns
        -------
        merged : (batch, seq_len, hidden_size)
        """
        cls_reps = torch.stack([s[:, 0, :] for s in streams], dim=1)  # (b, ns, h)
        scores = self.cwsa_linear(cls_reps).squeeze(-1)                # (b, ns)
        weights = F.softmax(scores, dim=-1)                             # (b, ns)
        weights_exp = weights.unsqueeze(-1).unsqueeze(-1)               # (b, ns, 1, 1)
        stacked = torch.stack(streams, dim=1)                           # (b, ns, seq, h)
        return (weights_exp * stacked).sum(dim=1)                       # (b, seq, h)

    def _apply_middle_block(
        self,
        mixed: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = mixed
        for layer in self.middle_layers:
            h = self._checkpoint_layer(layer, h, attention_mask)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        stream_snapshots: Dict[int, List[torch.Tensor]] = {}

        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids)

        for layer in self.begin_layers:
            hidden_states = self._checkpoint_layer(layer, hidden_states, attention_mask)

        # Initialise multi-stream phase
        streams: List[torch.Tensor] = [hidden_states.clone() for _ in range(self.num_streams)]
        merged: Optional[torch.Tensor] = None

        for loop_idx in range(self.num_middle_loops):
            # loop_idx is 0-indexed; merge_at is 1-indexed
            one_indexed = loop_idx + 1

            if self.enable_stream_snapshots:
                if merged is None:
                    snap = streams
                else:
                    snap = [merged]  # single stream post-merge
                if not self.training:
                    stream_snapshots[loop_idx] = [s.detach().cpu() for s in snap]
                else:
                    stream_snapshots[loop_idx] = [s.detach() for s in snap]

            loop_emb = self.loop_index_embeddings(
                torch.tensor(loop_idx, device=hidden_states.device, dtype=torch.long)
            )

            if merged is None:
                # Multi-stream phase: use depth/width projections
                concatenated = torch.cat(streams, dim=-1)
                mixed = self.depth_projs[loop_idx](concatenated)
                mixed = mixed + loop_emb.unsqueeze(0).unsqueeze(0)
                block_out = self._apply_middle_block(mixed, attention_mask)

                scattered = self.width_projs[loop_idx](block_out)
                scattered_per_stream = scattered.chunk(self.num_streams, dim=-1)
                streams = [streams[i] + scattered_per_stream[i] for i in range(self.num_streams)]

                # Merge at merge_at loop iteration (1-indexed)
                if one_indexed == self.merge_at:
                    merged = self._cwsa_merge(streams)
                    streams = []  # free memory
            else:
                # Single-stream phase: act like LoopedBERT
                merged = merged + loop_emb.unsqueeze(0).unsqueeze(0)
                merged = self._apply_middle_block(merged, attention_mask)

        # If merge never happened (merge_at > num_loops, shouldn't occur but be safe)
        if merged is None:
            merged = self._cwsa_merge(streams)

        # End block
        for layer in self.end_layers:
            merged = self._checkpoint_layer(layer, merged, attention_mask)

        pooler_output = self.pooler(merged)
        mlm_logits = self.mlm_head(merged)

        return {
            "last_hidden_state": merged,
            "pooler_output": pooler_output,
            "mlm_logits": mlm_logits,
            "stream_snapshots": stream_snapshots,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_ARCHITECTURE_MAP: Dict[str, type] = {
    "VanillaBERT": VanillaBERT,
    "LoopedBERT": LoopedBERT,
    "ALBERTLoopedBERT": ALBERTLoopedBERT,
    "HyperloopBERT": HyperloopBERT,
    "EarlyMergeHyperloopBERT": EarlyMergeHyperloopBERT,
}


def build_model(
    architecture: str,
    size: str = "base",
    **kwargs: Any,
) -> _BaseModel:
    """
    Instantiate a model by name.

    Parameters
    ----------
    architecture : str
        One of 'VanillaBERT', 'LoopedBERT', 'ALBERTLoopedBERT',
        'HyperloopBERT', 'EarlyMergeHyperloopBERT'.
    size : str
        One of 'tiny', 'small', 'base'.
    **kwargs :
        Additional keyword arguments passed to the model constructor
        (e.g. num_streams, merge_at, dropout, use_bf16).

    Returns
    -------
    model : _BaseModel subclass
    """
    if architecture not in _ARCHITECTURE_MAP:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from {list(_ARCHITECTURE_MAP)}."
        )
    cls = _ARCHITECTURE_MAP[architecture]
    return cls(size=size, **kwargs)

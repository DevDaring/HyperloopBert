import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import sys

from common.attention import force_full_precision_attention

# CITATION: Nangia, N. et al. (2020). CrowS-Pairs: A Challenge Dataset for
#           Measuring Social Biases in Masked Language Models. EMNLP 2020.
# CITATION: Khandelwal, K. et al. (2023). Indian-BhED: A Dataset for
#           Measuring India-Centric Bias in Large Language Models.
#           arXiv:2309.08573.
# CITATION: Zhao, J. et al. (2018). Gender Bias in Coreference Resolution:
#           Evaluation and Debiasing Methods (WinoBias). NAACL 2018.
# CITATION: Blodgett, S.L. et al. (2021). Stereotyping Norwegian Salmon:
#           An Inventory of Pitfalls in Fairness Benchmark Datasets. ACL 2021.
#           [PLL validity critique; motivation for SS-PLL]
# PRE-REGISTERED ENDPOINT: The primary comparison is at matched validation loss
#           (iso-perplexity), NOT at fixed token budget. This removes model quality
#           as an alternative explanation for bias differences.
# DATASET WARNING: These datasets contain stereotypical content by design.
#           Research/fairness-audit use only.

def _autocast_device_type(device) -> str:
    """torch.autocast requires 'cuda'/'cpu', not 'cuda:0' or a torch.device."""
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(':')[0]


def _fp32_forward(model, ac_device: str, **model_inputs):
    """
    Run one scoring forward in FULL FP32.

    All PLL/SS-PLL/WinoBias scoring forwards go through here: BF16 rounding is
    the same order as small PLL gaps on borderline pairs, so scoring under
    autocast would add metric noise exactly where the paired contrasts are
    decided. Autocast is explicitly disabled (in case a caller wrapped us) and
    attention is routed off the BF16-only flash kernel for the duration.
    """
    with force_full_precision_attention(), \
         torch.autocast(device_type=ac_device, enabled=False):
        return model(**model_inputs)


# Number of single-token-masked copies scored per forward pass. Bounds peak
# memory deterministically instead of relying on the CUDA OOM handler.
_PLL_CHUNK_SIZE: int = 32


def _get_logits(outputs):
    logits = outputs.get('mlm_logits', None)
    if logits is None:
        logits = outputs['logits'] if 'logits' in outputs else outputs[0]
    return logits


@torch.no_grad()
def compute_pll(model, tokenizer, sentence: str, device: str = 'cuda', max_length: int = 128) -> Optional[float]:
    """
    Compute Pseudo-Log-Likelihood for a sentence.
    For each token position, mask it, get the log probability of the correct token,
    accumulate, and normalize by the number of subword tokens (not counting [CLS] and [SEP]).
    Returns: float (PLL score, higher = model finds sentence more probable),
             or None if the sentence is degenerate or scoring failed (callers
             must treat None as missing, never as a valid score).
    """
    try:
        inputs = tokenizer(sentence, return_tensors="pt", max_length=max_length, truncation=True, padding=False)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        seq_len = input_ids.size(1)
        if seq_len <= 2: # Only special tokens -- no scoreable content
            return None

        total_log_prob = 0.0
        num_tokens = seq_len - 2 # Exclude [CLS] and [SEP]
        ac_device = _autocast_device_type(device)

        # Chunked batching: score up to _PLL_CHUNK_SIZE masked copies per forward
        for chunk_start in range(0, num_tokens, _PLL_CHUNK_SIZE):
            chunk_positions = list(range(chunk_start, min(chunk_start + _PLL_CHUNK_SIZE, num_tokens)))
            masked_input_ids = input_ids.repeat(len(chunk_positions), 1)
            for row, i in enumerate(chunk_positions):
                masked_input_ids[row, i + 1] = tokenizer.mask_token_id
            masked_attention_mask = attention_mask.repeat(len(chunk_positions), 1)

            outputs = _fp32_forward(model, ac_device,
                                    input_ids=masked_input_ids,
                                    attention_mask=masked_attention_mask)
            logits = _get_logits(outputs)

            for row, i in enumerate(chunk_positions):
                token_log_probs = F.log_softmax(logits[row, i + 1, :].float(), dim=-1)
                actual_token_id = input_ids[0, i + 1]
                total_log_prob += token_log_probs[actual_token_id].item()

        return total_log_prob / num_tokens
    except Exception as e:
        print(f"Error computing PLL: {e}", file=sys.stderr)
        return None


@torch.no_grad()
def compute_ss_pll(model, tokenizer, stereo_sentence: str, anti_sentence: str, device: str = 'cuda', max_length: int = 128) -> Tuple[Optional[float], Optional[float]]:
    """
    SS-PLL: Score only the shared (unmodified) tokens between stereo and anti sentences.
    Identifies shared token positions (same token in same position).
    Scores only those positions using PLL masking.
    Addresses Blodgett et al. 2021 surface-form critique.
    Returns: tuple (ss_pll_stereo, ss_pll_anti)
    """
    try:
        stereo_inputs = tokenizer(stereo_sentence, return_tensors="pt", max_length=max_length, truncation=True, padding=False)
        anti_inputs = tokenizer(anti_sentence, return_tensors="pt", max_length=max_length, truncation=True, padding=False)
        
        stereo_ids = stereo_inputs["input_ids"][0].tolist()
        anti_ids = anti_inputs["input_ids"][0].tolist()
        
        # Find shared indices (exact match in sequence)
        shared_indices_stereo = []
        shared_indices_anti = []
        
        # Simple longest common subsequence approach for exact matches could be used,
        # but CrowS-Pairs typically modifies a specific span.
        # We'll find tokens that appear in both, maintaining relative order, but a simpler
        # heuristic is finding identical tokens at identical offsets from ends, or just LCS.
        # Let's use a greedy match for simplicity, or just exact match for tokens that 
        # appear exactly once in both. For robustness, we'll find common prefix and suffix.
        
        # Common prefix
        prefix_len = 0
        while prefix_len < min(len(stereo_ids), len(anti_ids)) and stereo_ids[prefix_len] == anti_ids[prefix_len]:
            prefix_len += 1
            
        # Common suffix
        suffix_len = 0
        while suffix_len < min(len(stereo_ids) - prefix_len, len(anti_ids) - prefix_len) and \
              stereo_ids[len(stereo_ids) - 1 - suffix_len] == anti_ids[len(anti_ids) - 1 - suffix_len]:
            suffix_len += 1
            
        shared_indices_stereo = list(range(1, prefix_len)) + list(range(len(stereo_ids) - suffix_len, len(stereo_ids) - 1))
        shared_indices_anti = list(range(1, prefix_len)) + list(range(len(anti_ids) - suffix_len, len(anti_ids) - 1))
        
        # Remove [CLS] and [SEP] indices if present
        shared_indices_stereo = [i for i in shared_indices_stereo if i > 0 and i < len(stereo_ids) - 1]
        shared_indices_anti = [i for i in shared_indices_anti if i > 0 and i < len(anti_ids) - 1]

        if not shared_indices_stereo:
            return None, None

        def score_indices(input_ids, attention_mask, indices):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            total_log_prob = 0.0
            ac_device = _autocast_device_type(device)

            for chunk_start in range(0, len(indices), _PLL_CHUNK_SIZE):
                chunk = indices[chunk_start:chunk_start + _PLL_CHUNK_SIZE]
                masked_batch = input_ids.repeat(len(chunk), 1)
                for batch_idx, seq_idx in enumerate(chunk):
                    masked_batch[batch_idx, seq_idx] = tokenizer.mask_token_id
                masked_attn = attention_mask.repeat(len(chunk), 1)

                outputs = _fp32_forward(model, ac_device,
                                        input_ids=masked_batch,
                                        attention_mask=masked_attn)
                logits = _get_logits(outputs)

                for batch_idx, seq_idx in enumerate(chunk):
                    log_probs = F.log_softmax(logits[batch_idx, seq_idx, :].float(), dim=-1)
                    actual_token_id = input_ids[0, seq_idx]
                    total_log_prob += log_probs[actual_token_id].item()

            return total_log_prob / len(indices)

        ss_pll_stereo = score_indices(stereo_inputs["input_ids"], stereo_inputs["attention_mask"], shared_indices_stereo)
        ss_pll_anti = score_indices(anti_inputs["input_ids"], anti_inputs["attention_mask"], shared_indices_anti)
        
        return ss_pll_stereo, ss_pll_anti
        
    except Exception as e:
        print(f"Error computing SS-PLL: {e}", file=sys.stderr)
        return None, None


def score_bias_pair(model, tokenizer, stereo_sentence: str, anti_sentence: str, device: str = 'cuda', compute_ss: bool = False) -> Dict:
    """
    Score a (stereo, anti) pair.
    Returns dict with:
        PLL_Stereotypical, PLL_AntiStereotypical,
        SS_PLL_Stereotypical, SS_PLL_AntiStereotypical,
        Effect_Size (= PLL_stereo - PLL_anti),
        Stereotype_Preferred (1 if PLL_stereo > PLL_anti else 0)
    """
    pll_stereo = compute_pll(model, tokenizer, stereo_sentence, device)
    pll_anti = compute_pll(model, tokenizer, anti_sentence, device)
    
    ss_pll_stereo, ss_pll_anti = None, None
    if compute_ss:
        ss_pll_stereo, ss_pll_anti = compute_ss_pll(model, tokenizer, stereo_sentence, anti_sentence, device)
        
    if pll_stereo is None or pll_anti is None:
        return {
            'PLL_Stereotypical': None,
            'PLL_AntiStereotypical': None,
            'SS_PLL_Stereotypical': None,
            'SS_PLL_AntiStereotypical': None,
            'Effect_Size': None,
            'Stereotype_Preferred': None
        }
        
    effect_size = pll_stereo - pll_anti
    preferred = 1 if pll_stereo > pll_anti else 0
    
    return {
        'PLL_Stereotypical': pll_stereo,
        'PLL_AntiStereotypical': pll_anti,
        'SS_PLL_Stereotypical': ss_pll_stereo,
        'SS_PLL_AntiStereotypical': ss_pll_anti,
        'Effect_Size': effect_size,
        'Stereotype_Preferred': preferred
    }

# NOTE: the former PLL-pair `score_winobias` was removed: it silently returned
# 0.0 accuracies for unpreprocessed input. WinoBias is scored exclusively with
# the masked-pronoun protocol below (stereotype-consistency, Kurita-style --
# NOT coreference accuracy; name results columns accordingly in the paper).

# CITATION: Kurita, K. et al. (2019). Measuring Bias in Contextualized Word
#           Representations. 1st Workshop on Gender Bias in NLP, ACL.
#           [masked-pronoun probability protocol for MLM gender-bias scoring]
# CITATION: Zhao, J. et al. (2018). Gender Bias in Coreference Resolution
#           (WinoBias). NAACL 2018.  [pro/anti stereotype sentence sets]
_MASC_PRONOUNS = ["he", "him", "his"]
_FEM_PRONOUNS = ["she", "her", "hers"]


@torch.no_grad()
def score_winobias_masked_pronoun(model, tokenizer, df, device: str = 'cuda',
                                  max_length: int = 128,
                                  return_counts: bool = False):
    """
    Score a WinoBias split with the masked-pronoun protocol.

    df must have columns 'sentence' (full sentence text) and 'pronoun'
    (the gold gendered pronoun as it appears in the sentence).

    For each sentence: replace the pronoun occurrence with [MASK], and compare
    the total probability mass the model assigns to same-gender pronouns vs
    opposite-gender pronouns at the masked position. The item is 'correct'
    when the gold pronoun's gender receives more mass.

    Returns accuracy in [0, 1], or None if nothing could be scored.
    With return_counts=True, returns (correct, scored) instead -- used by the
    Stage 1 capability gate, whose binomial test needs raw counts.
    """
    ac_device = _autocast_device_type(device)
    masc_ids = [tokenizer.convert_tokens_to_ids(p) for p in _MASC_PRONOUNS]
    fem_ids = [tokenizer.convert_tokens_to_ids(p) for p in _FEM_PRONOUNS]
    unk = tokenizer.unk_token_id
    masc_ids = [i for i in masc_ids if i is not None and i != unk]
    fem_ids = [i for i in fem_ids if i is not None and i != unk]
    if not masc_ids or not fem_ids:
        return (0, 0) if return_counts else None

    correct = 0
    scored = 0
    for _, row in df.iterrows():
        sentence = row.get('sentence')
        pronoun = row.get('pronoun')
        if not isinstance(sentence, str) or not isinstance(pronoun, str):
            continue
        gold_is_masc = pronoun.lower() in _MASC_PRONOUNS
        if not gold_is_masc and pronoun.lower() not in _FEM_PRONOUNS:
            continue

        # Replace the first standalone occurrence of the pronoun with the mask token
        import re
        pattern = r'\b' + re.escape(pronoun) + r'\b'
        masked_sentence, n_sub = re.subn(pattern, tokenizer.mask_token, sentence, count=1)
        if n_sub == 0:
            continue

        inputs = tokenizer(masked_sentence, return_tensors="pt", max_length=max_length,
                           truncation=True, padding=False)
        input_ids = inputs["input_ids"].to(device)
        mask_positions = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if mask_positions.numel() == 0:
            continue
        pos = mask_positions[0].item()

        outputs = _fp32_forward(model, ac_device,
                                input_ids=input_ids,
                                attention_mask=inputs["attention_mask"].to(device))
        logits = _get_logits(outputs)
        probs = F.softmax(logits[0, pos, :].float(), dim=-1)
        masc_mass = probs[masc_ids].sum().item()
        fem_mass = probs[fem_ids].sum().item()

        pred_is_masc = masc_mass > fem_mass
        if pred_is_masc == gold_is_masc:
            correct += 1
        scored += 1

    if return_counts:
        return correct, scored
    if scored == 0:
        return None
    return correct / scored


def passes_quality_screen(pseudo_perplexity: float, threshold: float = 60.0) -> bool:
    """Returns True if model has learned enough to produce meaningful bias scores."""
    if pseudo_perplexity is None:
        return False
    return pseudo_perplexity <= threshold

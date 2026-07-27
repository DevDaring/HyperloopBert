"""
Stage3/stream_analysis_stage3.py -- the mechanistic suite.

1. Per-loop stream disagreement + correlation with bias effect size
   (CORRELATIONAL evidence: reported as corroborating, never causal).
2. Loop-wise representation similarity via linear CKA.
3. Early-merge OOD intervention: the TRAINED 4-stream HyperloopBERT weights
   are loaded into the EarlyMergeHyperloopBERT forward at merge_at in {1,2,3}
   -- an eval-time intervention, no training (spec framing: corroborating,
   out-of-distribution, not causal proof).
4. Demographic token drift across loop depths in stereotypical vs
   anti-stereotypical contexts.
5. Hyper-connection matrix statistics: learned depth/width projection norms,
   deviation from the structured init, and spectral norms per loop.
   CITATION: Zhu, D. et al. (2025). Hyper-Connections. ICLR 2025.
   CITATION: Xie, Z. et al. (2025). MHC: Manifold-Constrained
             Hyper-Connections. arXiv:2512.24880.  [unconstrained
             hyper-connections can destabilise at scale; these statistics
             report whether our scale shows that instability]
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import PreTrainedTokenizerFast
import argparse

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.architectures import build_model, get_model_info
from common.bias_metrics import score_bias_pair
from common.io_schemas import (
    EARLY_MERGE_COLUMNS, STREAM_DISAGREEMENT_COLUMNS, TOKEN_DRIFT_COLUMNS,
    REP_SIMILARITY_COLUMNS, HYPERCONNECTION_STATS_COLUMNS,
)
import Stage3.config_stage3 as cfg
from Stage1.eval_bias_stage1 import extract_model_metadata
from Stage2.loop_trajectory_stage2 import extract_intermediate_representations

logger = setup_logging('stream_analysis_stage3')

# Curated demographic terms per category for the token-drift analysis
# (single-wordpiece-friendly, high-frequency terms)
DEMOGRAPHIC_TERMS = {
    'gender': ['man', 'woman', 'he', 'she', 'his', 'her'],
    'race': ['black', 'white', 'asian', 'african', 'european'],
    'religion': ['muslim', 'christian', 'jewish', 'hindu'],
    'age': ['old', 'young', 'elderly'],
    'nationality': ['american', 'indian', 'mexican', 'chinese'],
}


def _identity(meta, model):
    info = get_model_info(model)
    return {
        'Stage': cfg.STAGE,
        'Architecture': meta['Architecture'],
        'Model_Size': meta['Model_Size'],
        'Hidden_Size': model.hidden_size,
        'Seed': meta['Seed'],
        'Unique_Parameters': info['Unique_Parameters'],
        'Total_Parameters': info['Total_Parameters'],
        'Effective_Depth': info['Effective_Depth'],
        'Shared_Ratio': info['Shared_Ratio'],
    }


def _append_rows(rows, columns, out_path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    header = not os.path.exists(out_path)
    df.to_csv(out_path, mode='a' if not header else 'w', header=header, index=False)


def analyze_stream_disagreement(model, tokenizer, df, device, meta, out_path):
    """
    PER-LOOP stream disagreement (mean pairwise cosine distance between the
    streams' [CLS] representations) correlated with the pair's bias effect
    size. One summary row per loop depth; per-item correlations attached.
    """
    logger.info("Analyzing stream disagreement (per loop)...")

    if meta['Architecture'] != 'HyperloopBERT':
        return

    model.enable_stream_snapshots = True
    per_loop = {}  # loop_key -> list of (disagreement, effect_size)

    for idx, row in df.iterrows():
        stereo, anti = row['stereo'], row['anti']
        try:
            inputs = tokenizer(stereo, return_tensors='pt', max_length=cfg.SEQ_LENGTH,
                               truncation=True).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=str(device).split(':')[0], dtype=torch.bfloat16):
                    out = model(**inputs)
            snapshots = out.get('stream_snapshots', {})
            if not snapshots:
                continue

            scores = score_bias_pair(model, tokenizer, stereo, anti, device)
            if scores['Effect_Size'] is None:
                continue

            for loop_key, streams in snapshots.items():
                if len(streams) < 2:
                    continue
                cls_streams = torch.stack([s[0, 0, :] for s in streams], dim=0).float()
                cos_sim = F.cosine_similarity(cls_streams.unsqueeze(1),
                                              cls_streams.unsqueeze(0), dim=-1)
                n = cos_sim.size(0)
                off_diag = cos_sim[~torch.eye(n, dtype=torch.bool, device=cos_sim.device)]
                disagreement = (1.0 - off_diag.mean()).item() if off_diag.numel() > 0 else 0.0
                per_loop.setdefault(loop_key, []).append((disagreement, scores['Effect_Size']))
        except Exception as e:
            logger.debug(f"Row {idx} skipped: {e}")

    model.enable_stream_snapshots = False

    from scipy.stats import pearsonr, spearmanr
    rows = []
    for loop_key in sorted(per_loop):
        pairs = per_loop[loop_key]
        if len(pairs) < 3:
            continue
        dis = [p[0] for p in pairs]
        eff = [p[1] for p in pairs]
        pr, pp = pearsonr(dis, eff)
        sr, sp = spearmanr(dis, eff)
        row = _identity(meta, model)
        row.update({
            'Loop_Depth': loop_key,
            'Stream_Disagreement': float(np.mean(dis)),
            'Effect_Size': float(np.mean(eff)),
            'Pearson_R': float(pr), 'Pearson_P': float(pp),
            'Spearman_R': float(sr), 'Spearman_P': float(sp),
            'Timestamp': datetime.utcnow().isoformat() + 'Z',
        })
        rows.append(row)
        logger.info(f"Loop {loop_key}: disagreement={np.mean(dis):.4f}, "
                    f"signed r={pr:.3f} (p={pp:.4f}) [CORRELATIONAL]")
    _append_rows(rows, STREAM_DISAGREEMENT_COLUMNS, out_path)


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between feature matrices (n_samples, dim), columns centered.
    CITATION: Kornblith, S. et al. (2019). Similarity of Neural Network
    Representations Revisited. ICML."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xty = y.T @ x
    num = np.linalg.norm(xty, ord='fro') ** 2
    den = (np.linalg.norm(x.T @ x, ord='fro') * np.linalg.norm(y.T @ y, ord='fro'))
    return float(num / den) if den > 0 else 0.0


def analyze_cka(model, tokenizer, df, device, meta, out_path):
    """
    Loop-wise representation similarity: linear CKA between [CLS] features at
    every pair of hooked depths, over a sample of evaluation sentences.
    Tests whether stream diversity prevents premature representational
    convergence across loop iterations.
    """
    logger.info("Computing loop-wise CKA...")
    feats = {}  # depth -> list of cls vectors

    for idx, row in df.iterrows():
        try:
            inputs = tokenizer(row['stereo'], return_tensors='pt',
                               max_length=cfg.SEQ_LENGTH, truncation=True).to(device)
            reps = extract_intermediate_representations(
                model, inputs['input_ids'], inputs['attention_mask'],
                meta['Architecture'])
            for depth, rep in reps.items():
                feats.setdefault(depth, []).append(rep[0, 0, :].float().cpu().numpy())
        except Exception as e:
            logger.debug(f"CKA row {idx} skipped: {e}")

    depths = sorted(feats)
    if len(depths) < 2:
        logger.warning("CKA: fewer than 2 hooked depths; nothing to compare.")
        return

    mats = {d: np.stack(feats[d]) for d in depths}
    rows = []
    for i, d1 in enumerate(depths):
        for d2 in depths[i + 1:]:
            n = min(len(mats[d1]), len(mats[d2]))
            cka = _linear_cka(mats[d1][:n], mats[d2][:n])
            row = _identity(meta, model)
            row.update({
                'Loop_Pair': f"{d1}-{d2}",
                'CKA': cka,
                'Timestamp': datetime.utcnow().isoformat() + 'Z',
            })
            rows.append(row)
    _append_rows(rows, REP_SIMILARITY_COLUMNS, out_path)
    logger.info(f"CKA: wrote {len(rows)} depth-pair rows.")


def early_merge_intervention(state_dict, tokenizer, df, device, meta, out_path):
    """
    Early-merge OOD intervention (spec: NO new training). The trained
    4-stream HyperloopBERT weights are loaded into EarlyMergeHyperloopBERT
    at each merge_at in cfg.EARLY_MERGE_POINTS; the same parameters are
    executed with streams merged early. Out-of-distribution for the weights
    -- corroborating evidence only, NEVER causal proof.
    """
    for merge_at in cfg.EARLY_MERGE_POINTS:
        logger.info(f"Early-merge intervention: merge_at={merge_at} (OOD, eval-time)...")
        em_model = build_model('EarlyMergeHyperloopBERT', meta['Model_Size'],
                               num_streams=cfg.DEFAULT_NUM_STREAMS, merge_at=merge_at)
        try:
            em_model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.error(f"Early-merge weight transfer failed (merge_at={merge_at}): {e}")
            continue
        em_model.to(device)
        em_model.eval()

        prefs, effects = [], []
        for idx, row in df.iterrows():
            scores = score_bias_pair(em_model, tokenizer, row['stereo'], row['anti'], device)
            if scores['Stereotype_Preferred'] is not None:
                prefs.append(scores['Stereotype_Preferred'])
                effects.append(scores['Effect_Size'])

        if not prefs:
            logger.warning(f"Early-merge merge_at={merge_at}: no pairs scored.")
            continue

        row_out = _identity(meta, em_model)
        row_out.update({
            'Merge_At': merge_at,
            'Overall_Stereotype_Preference_Rate': float(np.mean(prefs)),
            'Mean_Effect_Size': float(np.mean(effects)),
            'Timestamp': datetime.utcnow().isoformat() + 'Z',
        })
        _append_rows([row_out], EARLY_MERGE_COLUMNS, out_path)
        logger.info(f"merge_at={merge_at}: preference={np.mean(prefs):.4f} "
                    f"(n={len(prefs)}) [OOD INTERVENTION]")
        del em_model
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()


def analyze_token_drift(model, tokenizer, df, device, meta, out_path):
    """
    Demographic token drift: for each curated demographic term appearing in a
    stereo/anti sentence, the cosine drift (1 - cos) of that token's hidden
    state between consecutive hooked depths. Compares how demographic-token
    representations move through the loops in stereotypical vs
    anti-stereotypical contexts.
    """
    logger.info("Analyzing demographic token drift...")
    rows = []

    for category, terms in DEMOGRAPHIC_TERMS.items():
        term_ids = {}
        for t in terms:
            tid = tokenizer.convert_tokens_to_ids(t)
            if tid is not None and tid != tokenizer.unk_token_id:
                term_ids[t] = tid
        if not term_ids:
            continue

        for context_type, col in (('stereotypical', 'stereo'), ('anti-stereotypical', 'anti')):
            drift_acc = {}  # (term, depth_pair) -> list of drifts
            for idx, row in df.iterrows():
                sentence = row[col]
                if not isinstance(sentence, str):
                    continue
                lowered = f" {sentence.lower()} "
                present = [t for t in term_ids if f" {t} " in lowered]
                if not present:
                    continue
                try:
                    inputs = tokenizer(sentence, return_tensors='pt',
                                       max_length=cfg.SEQ_LENGTH, truncation=True).to(device)
                    ids = inputs['input_ids'][0].tolist()
                    reps = extract_intermediate_representations(
                        model, inputs['input_ids'], inputs['attention_mask'],
                        meta['Architecture'])
                    depths = sorted(reps)
                    for t in present:
                        if term_ids[t] not in ids:
                            continue
                        pos = ids.index(term_ids[t])
                        for d1, d2 in zip(depths, depths[1:]):
                            v1 = reps[d1][0, pos, :].float()
                            v2 = reps[d2][0, pos, :].float()
                            drift = 1.0 - F.cosine_similarity(v1, v2, dim=0).item()
                            drift_acc.setdefault((t, d2), []).append(drift)
                except Exception as e:
                    logger.debug(f"Token drift row {idx} skipped: {e}")

            for (term, depth), drifts in drift_acc.items():
                row_out = _identity(meta, model)
                row_out.update({
                    'Category': category,
                    'Demographic_Term': term,
                    'Context_Type': context_type,
                    'Loop_Depth': depth,
                    'Cosine_Drift': float(np.mean(drifts)),
                    'Timestamp': datetime.utcnow().isoformat() + 'Z',
                })
                rows.append(row_out)

    _append_rows(rows, TOKEN_DRIFT_COLUMNS, out_path)
    logger.info(f"Token drift: wrote {len(rows)} rows.")


@torch.no_grad()
def analyze_hyperconnection_stats(model, meta, out_path):
    """
    Hyper-connection matrix statistics (no forwards, weights only).

    Xie et al. 2025 (MHC, arXiv:2512.24880) show unconstrained hyper-connection
    matrices can destabilise at scale. These statistics let the paper report
    whether the learned depth/width projections stayed near their structured
    init (depth block ~ I/n, width block ~ 0.5*I) or drifted: per-stream block
    Frobenius norms, deviation from init, and the full matrix spectral norm
    (a growth factor > 1 per loop compounds over the 4 iterations).
    """
    logger.info("Computing hyper-connection matrix statistics...")
    if not hasattr(model, 'depth_projs') or not hasattr(model, 'width_projs'):
        return

    h = model.hidden_size
    n = model.num_streams
    eye = torch.eye(h)
    rows = []

    def _block_rows(loop_idx, proj_name, weight, blocks, init_block):
        spectral = float(torch.linalg.matrix_norm(weight.float(), ord=2))
        for s, block in enumerate(blocks):
            row = _identity(meta, model)
            row.update({
                'Loop_Index': loop_idx,
                'Projection': proj_name,
                'Stream_Index': s,
                'Block_Frobenius_Norm': float(torch.linalg.matrix_norm(block, ord='fro')),
                'Block_Deviation_From_Init': float(torch.linalg.matrix_norm(block - init_block, ord='fro')),
                'Matrix_Spectral_Norm': spectral,
                'Timestamp': datetime.utcnow().isoformat() + 'Z',
            })
            rows.append(row)

    for loop_idx, (depth_proj, width_proj) in enumerate(zip(model.depth_projs,
                                                            model.width_projs)):
        dw = depth_proj.weight.detach().float().cpu()          # (h, n*h)
        ww = width_proj.weight.detach().float().cpu()          # (n*h, h)
        _block_rows(loop_idx, 'depth_proj',
                    dw, [dw[:, s * h:(s + 1) * h] for s in range(n)], eye / n)
        _block_rows(loop_idx, 'width_proj',
                    ww, [ww[s * h:(s + 1) * h, :] for s in range(n)], eye * 0.5)

    _append_rows(rows, HYPERCONNECTION_STATS_COLUMNS, out_path)
    max_spec = max(r['Matrix_Spectral_Norm'] for r in rows) if rows else float('nan')
    logger.info(f"Hyper-connection stats: {len(rows)} rows; "
                f"max spectral norm {max_spec:.3f} "
                f"(report vs the MHC instability criterion in the paper).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
    parser.add_argument('--sample', type=int, default=100,
                        help='Number of pairs sampled for the mechanistic analyses')
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage3.py directly instead.")
        return

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, cfg.RESULTS_DIR)
    models_dir = os.path.join(base_dir, cfg.MODELS_DIR)

    tokenizer_dir = os.path.join(data_dir, 'tokenizer')
    if not os.path.exists(tokenizer_dir):
        logger.error(f"Tokenizer not found at {tokenizer_dir}")
        return
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    eval_dir = os.path.join(data_dir, 'datasets_eval')
    multicrows_path = os.path.join(eval_dir, 'multicrows', 'crows_pair_english.csv')
    if not os.path.exists(multicrows_path):
        logger.error("Multi-CrowS-Pairs not found.")
        return

    df = pd.read_csv(multicrows_path)
    if len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # One snapshot per (arch, seed): the deepest crossed band for the
    # PRIMARY 4-stream HyperloopBERT and for LoopedBERT (drift comparison)
    candidates = {}
    type_dir = os.path.join(models_dir, 'iso_band_models')
    if os.path.exists(type_dir):
        for root, dirs, files in os.walk(type_dir):
            if 'pytorch_model.bin' not in files:
                continue
            meta = extract_model_metadata(root)
            if not meta or meta['Band'] is None:
                continue
            if meta.get('Merge_At') is not None:
                continue
            if meta['Architecture'] == 'HyperloopBERT' and meta.get('Stream_Count') not in (None, cfg.DEFAULT_NUM_STREAMS):
                continue
            if meta['Architecture'] not in ('HyperloopBERT', 'LoopedBERT'):
                continue
            key = (meta['Architecture'], meta['Seed'])
            if key not in candidates or meta['Band'] < candidates[key][0]:
                candidates[key] = (meta['Band'], root, meta)

    disagreement_out = os.path.join(results_dir, 'mechanistic', 'stream_disagreement.csv')
    cka_out = os.path.join(results_dir, 'mechanistic', 'representation_similarity.csv')
    early_merge_out = os.path.join(results_dir, 'mechanistic', 'early_merge_intervention.csv')
    drift_out = os.path.join(results_dir, 'mechanistic', 'demographic_token_drift.csv')
    hc_stats_out = os.path.join(results_dir, 'mechanistic', 'hyperconnection_stats.csv')

    for (arch, seed), (band, cp_dir, meta) in candidates.items():
        logger.info(f"Mechanistic suite on {arch} seed {seed} (band {band})...")
        kwargs = {'num_streams': cfg.DEFAULT_NUM_STREAMS} if arch == 'HyperloopBERT' else {}
        model = build_model(arch, meta['Model_Size'], **kwargs)
        try:
            state = torch.load(os.path.join(cp_dir, 'pytorch_model.bin'),
                               map_location='cpu', weights_only=True)
            model.load_state_dict(state)
        except Exception as e:
            logger.error(f"Failed to load {cp_dir}: {e}")
            continue
        model.to(device)
        model.eval()

        if arch == 'HyperloopBERT':
            analyze_hyperconnection_stats(model, meta, hc_stats_out)
            analyze_stream_disagreement(model, tokenizer, df, device, meta, disagreement_out)
            early_merge_intervention(state, tokenizer, df, device, meta, early_merge_out)
        analyze_cka(model, tokenizer, df, device, meta, cka_out)
        analyze_token_drift(model, tokenizer, df, device, meta, drift_out)

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

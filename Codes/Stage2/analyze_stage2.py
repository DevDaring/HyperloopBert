import os
import sys
import glob as globmod
import pandas as pd
import argparse
from datetime import datetime

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.iso_loss import compute_primary_band
from common.stats_engine import holm_bonferroni_correction
from common.plotting import plot_iso_loss_bias, plot_loop_trajectory, plot_pareto_front
from Stage1.analyze_stage1 import (
    load_data, compute_contrast, compute_item_level_contrast, _load_item_frame,
)
import Stage2.config_stage2 as cfg

logger = setup_logging('analyze_stage2')

# PRE-REGISTERED CONFIRMATORY FAMILY (Holm-Bonferroni corrected, item-level
# paired permutation as the primary test): Vanilla vs Looped, Vanilla vs
# ALBERT, Looped vs ALBERT -- Multi-CrowS-Pairs PLL Effect_Size at the primary
# iso-loss band, base size. Seed-level contrasts are robustness checks only.

CONFIRMATORY_PAIRS = [
    ('VanillaBERT', 'LoopedBERT'),
    ('VanillaBERT', 'ALBERTLoopedBERT'),
    ('LoopedBERT', 'ALBERTLoopedBERT'),
]


def load_stage2_data(results_dir):
    mlm_df, bias_df, indian_df = load_data(results_dir)

    wino_path = os.path.join(results_dir, 'bias', 'winobias_summary.csv')
    glue_path = os.path.join(results_dir, 'glue', 'summary_table.csv')
    calib_path = os.path.join(results_dir, 'bias', 'external_calibration.csv')
    traj_path = os.path.join(results_dir, 'mechanistic', 'loop_trajectory.csv')

    wino_df = pd.read_csv(wino_path) if os.path.exists(wino_path) else None
    glue_df = pd.read_csv(glue_path) if os.path.exists(glue_path) else None
    calib_df = pd.read_csv(calib_path) if os.path.exists(calib_path) else None
    traj_df = pd.read_csv(traj_path) if os.path.exists(traj_path) else None

    return mlm_df, bias_df, indian_df, wino_df, glue_df, calib_df, traj_df


def ss_pll_direction_agreement(results_dir, band):
    """
    SS-PLL validity check on the PRIMARY CONTRAST (spec 9.1): does the
    shared-token metric agree in DIRECTION with full-sentence PLL for
    Vanilla vs Looped at the primary band (base size)?
    Returns (agrees: bool or None, detail: str).
    """
    def item_ss_means(arch):
        pattern = os.path.join(results_dir, 'bias',
                               f"multicrows_{arch}_base_seed*_band{band}_progress.csv")
        frames = []
        for path in globmod.glob(pattern):
            if '_streams' in os.path.basename(path):
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            need = ['Row_Index', 'SS_PLL_Stereotypical', 'SS_PLL_AntiStereotypical']
            if not all(c in df.columns for c in need):
                continue
            df = df.dropna(subset=need[1:])
            if df.empty:
                continue
            df['SS_Delta'] = df['SS_PLL_Stereotypical'] - df['SS_PLL_AntiStereotypical']
            frames.append(df[['Row_Index', 'SS_Delta']])
        if not frames:
            return None
        allf = pd.concat(frames, ignore_index=True)
        return allf.groupby('Row_Index')['SS_Delta'].mean()

    v = item_ss_means('VanillaBERT')
    l = item_ss_means('LoopedBERT')
    if v is None or l is None:
        return None, "SS-PLL data unavailable for the primary contrast"
    common = v.index.intersection(l.index)
    if len(common) < 10:
        return None, "too few shared items for SS-PLL agreement"
    ss_delta = float((v.loc[common] - l.loc[common]).mean())
    detail = f"SS-PLL primary-contrast delta = {ss_delta:.4f}"
    return ss_delta > 0, detail  # PLL predicts Vanilla > Looped (positive)


def calibration_check(calib_df):
    """
    External-calibration sanity (spec 9.1): both public anchors present, and
    both show above-chance stereotype preference on the CANONICAL shared-token
    metric (the one comparable to published CrowS-Pairs numbers). The observed
    anchor ordering is recorded for the paper but does NOT gate: our PLL
    variant differs from published scoring, so demanding the published
    ordering with a different metric would be an invalid check.
    """
    anchors = ['bert-base-uncased', 'albert-base-v2']
    if calib_df is None or calib_df.empty:
        return False, "no calibration results"
    rows = calib_df[calib_df['Dataset'] == 'multicrows'] if 'Dataset' in calib_df.columns else calib_df
    present = {}
    for anchor in anchors:
        sub = rows[rows['Model_Name'] == anchor]
        if sub.empty:
            return False, f"anchor {anchor} missing from calibration results"
        col = ('Shared_Token_Preference_Rate'
               if 'Shared_Token_Preference_Rate' in sub.columns
               and sub['Shared_Token_Preference_Rate'].notna().any()
               else 'Overall_Stereotype_Preference_Rate')
        present[anchor] = float(sub[col].iloc[-1])
    sane = all(v > 0.5 for v in present.values())
    ordering = ' >= '.join(sorted(present, key=present.get, reverse=True))
    detail = (f"anchors {present} (shared-token metric); observed ordering: {ordering}; "
              f"above-chance: {sane}")
    return sane, detail


def glue_quality_screen(glue_df, threshold):
    """
    Return the set of (Architecture, Model_Size, Seed) whose GLUE average is
    below the threshold (spec 9.1: excluded from confirmatory contrasts,
    logged -- never silently).
    """
    excluded = set()
    if glue_df is None or glue_df.empty:
        logger.warning("GLUE screen: no GLUE results found; screen not applied.")
        return excluded
    group_cols = ['Architecture', 'Model_Size', 'Seed']
    agg = glue_df.groupby(group_cols)['Accuracy'].mean().reset_index()
    for _, row in agg.iterrows():
        if row['Accuracy'] * 100.0 < threshold:
            key = (row['Architecture'], row['Model_Size'], row['Seed'])
            excluded.add(key)
            logger.warning(f"GLUE screen: excluding {key} "
                           f"(avg accuracy {row['Accuracy']*100:.1f}% < {threshold}%).")
    return excluded


def decision_logic(confirmatory, ss_agrees, ss_detail, calib_sane, calib_detail):
    """
    GO: Vanilla > Looped and Vanilla > ALBERT directions hold, the V-vs-L
        item-level contrast survives Holm, SS-PLL agrees in direction on the
        primary contrast, and calibration anchors are sane.
    PAUSE: effect present but a validity leg fails.
    NO-GO: direction reverses on the primary contrast.
    """
    by_name = {c['Contrast']: c for c in confirmatory}
    v_vs_l = by_name.get('VanillaBERT_vs_LoopedBERT')
    v_vs_a = by_name.get('VanillaBERT_vs_ALBERTLoopedBERT')

    if v_vs_l is None or v_vs_a is None:
        return "PAUSE (confirmatory contrasts incomplete - check eval coverage)"

    if v_vs_l['Mean_Delta'] < -0.0:
        return "NO-GO (primary contrast direction reversed: Looped > Vanilla)"

    v_gt_l = v_vs_l['Mean_Delta'] > 0
    v_gt_a = v_vs_a['Mean_Delta'] > 0
    v_l_sig = bool(v_vs_l.get('Significant'))

    if not (v_gt_l and v_gt_a):
        return "PAUSE (sharing direction not consistent across the spectrum)"
    if not v_l_sig:
        return "PAUSE (primary contrast does not survive Holm correction)"
    if ss_agrees is False:
        return f"PAUSE (SS-PLL disagrees in direction on the primary contrast: {ss_detail})"
    if ss_agrees is None:
        return f"PAUSE (SS-PLL agreement not computable: {ss_detail})"
    if not calib_sane:
        return f"PAUSE (external calibration failed: {calib_detail})"
    return "GO"


def write_paper_outline(results_dir, confirmatory, decision, ss_detail, calib_detail):
    outline_path = os.path.join(results_dir, 'stage2_paper_outline.md')
    is_go = decision.startswith('GO')

    if is_go:
        abstract = ("Expanding on Stage 1, this stage isolates weight sharing from "
                    "architectural depth by comparing VanillaBERT, LoopedBERT, and "
                    "ALBERTLoopedBERT at matched validation loss. Bias decreases with "
                    "the degree of cross-layer parameter sharing, and the effect "
                    "survives shared-token (SS-PLL) scoring and Holm correction.")
    else:
        abstract = "[NOT ASSERTED -- verdict was not GO.]"

    outline = f"""# Stage 2 Paper Outline

## Title
Structural Weight Sharing is the Locus of Bias Reduction in Looped Transformers

## Finding-First Abstract
{abstract}

## Verdict
**{decision}**

## Validity legs
- SS-PLL: {ss_detail}
- Calibration: {calib_detail}

### Confirmatory family (item-level primary, Holm corrected):
"""
    for c in confirmatory:
        outline += (f"- **{c['Contrast']}**: Delta = {c['Mean_Delta']:.4f} "
                    f"(p_raw = {c['P_Value']:.4f}, p_holm = {c['Holm_Corrected_P_Value']:.4f}, "
                    f"significant = {c['Significant']}, n_items = {c.get('N_Items')})\n")

    outline += """
## Limitations
- 400M-token budget; models are architecture probes, not SOTA encoders
- GLUE average trails full BERT-base (screen threshold documented)
"""
    os.makedirs(os.path.dirname(outline_path), exist_ok=True)
    with open(outline_path, 'w') as f:
        f.write(outline)

    logger.info(f"Paper outline written to {outline_path}")


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(base_dir, cfg.RESULTS_DIR)
    figures_dir = os.path.join(base_dir, cfg.FIGURES_DIR)

    mlm_df, bias_df, indian_df, wino_df, glue_df, calib_df, traj_df = load_stage2_data(results_dir)

    if mlm_df is None or bias_df is None:
        logger.error("Required data not found. Run evaluation first.")
        return

    # GLUE quality screen (exclusions logged, applied to seed-level contrasts)
    excluded = glue_quality_screen(glue_df, getattr(cfg, 'GLUE_QUALITY_SCREEN', 55.0))
    if excluded and not bias_df.empty:
        mask = bias_df.apply(lambda r: (r['Architecture'], r['Model_Size'], r['Seed']) in excluded, axis=1)
        bias_df = bias_df[~mask]

    # Primary band at base size (per-seed coverage enforced)
    primary_bands = {}
    for size in ['base']:
        band = compute_primary_band(mlm_df, cfg.ARCHITECTURES, size, logger_obj=logger)
        primary_bands[size] = band
        logger.info(f"Primary band for size {size}: {band}")
    band = primary_bands.get('base')

    metric = 'Overall_Stereotype_Preference_Rate'

    # Confirmatory family: item-level primary + seed-level robustness
    confirmatory = []
    for arch1, arch2 in CONFIRMATORY_PAIRS:
        name = f"{arch1}_vs_{arch2}"
        item = (compute_item_level_contrast(results_dir, 'multicrows', arch1, arch2,
                                            'base', band)
                if band is not None else None)
        seedc = compute_contrast(bias_df, arch1, arch2, metric, primary_bands).get('base')
        if item is None:
            logger.warning(f"{name}: item-level contrast not computable.")
            continue
        confirmatory.append({
            'Contrast': name,
            'Metric': 'Effect_Size (item-level)',
            'Dataset': 'Multi-CrowS-Pairs',
            'Band': band,
            'Model_Size': 'base',
            'Mean_Delta': item['mean_delta'],
            'P_Value': item['p_value'],
            'CI_Low': item['ci_low'],
            'CI_High': item['ci_high'],
            'Cohens_D': item['cohens_d'],
            'N_Items': item['n_items'],
            'Preference_Rate_Delta': item['preference_rate_delta'],
            'Seed_Level_Delta': seedc['Mean_Delta'] if seedc else None,
            'Seed_Level_P': seedc['P_Value'] if seedc else None,
            'Mean_Loss_Gap': seedc['Mean_Loss_Gap'] if seedc else None,
            'Max_Loss_Gap': seedc['Max_Loss_Gap'] if seedc else None,
        })

    # Holm-Bonferroni over the item-level p-values
    p_values = [(c['Contrast'], c['P_Value']) for c in confirmatory]
    corrected = holm_bonferroni_correction(p_values)
    cmap = {name: (corr, sig) for name, _, corr, sig in corrected}
    for c in confirmatory:
        corr, sig = cmap.get(c['Contrast'], (None, False))
        c['Holm_Corrected_P_Value'] = corr
        c['Significant'] = sig

    # Validity legs
    ss_agrees, ss_detail = (ss_pll_direction_agreement(results_dir, band)
                            if band is not None else (None, 'no primary band'))
    calib_sane, calib_detail = calibration_check(calib_df)
    logger.info(f"SS-PLL agreement: {ss_agrees} ({ss_detail})")
    logger.info(f"Calibration: {calib_sane} ({calib_detail})")

    # Decision
    decision = decision_logic(confirmatory, ss_agrees, ss_detail, calib_sane, calib_detail)
    logger.info(f"=== GO/PAUSE VERDICT: {decision} ===")
    for c in confirmatory:
        logger.info(f"{c['Contrast']}: delta={c['Mean_Delta']:.4f}, "
                    f"p_holm={c['Holm_Corrected_P_Value']}")

    # Exploratory: parameter-matched Vanilla control (never confirmatory)
    exploratory = []
    if band is not None:
        pm = compute_item_level_contrast(results_dir, 'multicrows',
                                         'VanillaBERT6', 'LoopedBERT', 'base', band,
                                         alternative='two-sided')
        if pm is not None:
            exploratory.append({
                'Contrast': 'VanillaBERT6_vs_LoopedBERT (parameter-matched control)',
                'Metric': 'Effect_Size (item-level)',
                'Dataset': 'Multi-CrowS-Pairs',
                'Band': band, 'Model_Size': 'base',
                'Mean_Delta': pm['mean_delta'], 'P_Value': pm['p_value'],
                'CI_Low': pm['ci_low'], 'CI_High': pm['ci_high'],
                'Cohens_D': pm['cohens_d'], 'N_Items': pm['n_items'],
                'Note': 'EXPLORATORY: uncorrected; parameter-matched, not compute-matched',
            })

    # Save stats with the REGISTERED filenames
    stats_dir = os.path.join(results_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    if confirmatory:
        pd.DataFrame(confirmatory).to_csv(os.path.join(stats_dir, 'confirmatory_family.csv'), index=False)
    if exploratory:
        pd.DataFrame(exploratory).to_csv(os.path.join(stats_dir, 'exploratory_results.csv'), index=False)

    # Outline
    write_paper_outline(results_dir, confirmatory, decision, ss_detail, calib_detail)

    # Figures
    for size in cfg.SIZES:
        plot_iso_loss_bias(bias_df, 'Multi-CrowS-Pairs', metric, size,
                           os.path.join(figures_dir, f'iso_loss_multicrows_{size}.png'))

    if traj_df is not None:
        plot_loop_trajectory(traj_df, 'Multi-CrowS-Pairs', os.path.join(figures_dir, 'loop_trajectory.png'))

    # Pareto (GLUE schema now carries Band)
    if glue_df is not None and not glue_df.empty and 'Band' in glue_df.columns:
        glue_agg = glue_df.groupby(['Architecture', 'Model_Size', 'Seed', 'Band'])['Accuracy'].mean().reset_index()
        glue_agg.rename(columns={'Accuracy': 'GLUE_Average'}, inplace=True)
        bias_agg = bias_df.groupby(['Architecture', 'Model_Size', 'Seed', 'Band'])[metric].mean().reset_index()
        pareto_df = pd.merge(glue_agg, bias_agg, on=['Architecture', 'Model_Size', 'Seed', 'Band'])
        if not pareto_df.empty:
            plot_pareto_front(pareto_df, metric, os.path.join(figures_dir, 'pareto_front.png'))

    logger.info("Stage 2 Analysis complete.")


if __name__ == "__main__":
    main()

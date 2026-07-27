import os
import sys
import pandas as pd
import argparse
from datetime import datetime

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.iso_loss import compute_primary_band
from common.stats_engine import holm_bonferroni_correction
from common.plotting import plot_stream_ablation, plot_stream_disagreement, plot_iso_loss_bias
from Stage1.analyze_stage1 import load_data, compute_contrast, compute_item_level_contrast
import Stage3.config_stage3 as cfg

logger = setup_logging('analyze_stage3')

# PRE-REGISTERED CONFIRMATORY FAMILY (Holm-Bonferroni, item-level primary):
#   1. Vanilla vs Hyperloop (PRIMARY)      -- expected delta > 0
#   2. Vanilla vs Looped                   -- expected delta > 0
#   3. Looped vs Hyperloop                 -- direction under test
#   4. Hyperloop n=4 vs n=1                -- expected delta < 0
# Everything else (early-merge, stream-disagreement correlation, per-category)
# is EXPLORATORY / corroborating. The stream-disagreement correlation is a
# CORRELATIONAL statistic: it may support the mechanistic narrative but must
# NEVER gate or license the word "causal" -- only the from-scratch stream
# dose-response is a causal manipulation.

MONOTONE_TOLERANCE = 0.02   # tolerance for n1 <= n2 <= n4 monotonicity
SANITY_TOLERANCE = 0.03     # |Hyperloop(n=1) - Looped| tolerance (approximate
                            # collapse: n=1 keeps its projections)


def load_stage3_data(results_dir):
    mlm_df, bias_df, indian_df = load_data(results_dir)

    glue_path = os.path.join(results_dir, 'glue', 'summary_table.csv')
    disagreement_path = os.path.join(results_dir, 'mechanistic', 'stream_disagreement.csv')
    early_merge_path = os.path.join(results_dir, 'mechanistic', 'early_merge_intervention.csv')

    glue_df = pd.read_csv(glue_path) if os.path.exists(glue_path) else None
    disagreement_df = pd.read_csv(disagreement_path) if os.path.exists(disagreement_path) else None
    early_merge_df = pd.read_csv(early_merge_path) if os.path.exists(early_merge_path) else None

    return mlm_df, bias_df, indian_df, glue_df, disagreement_df, early_merge_df


def stream_rate(bias_df, band, n):
    """Mean preference rate for HyperloopBERT with Stream_Count == n at band."""
    if 'Stream_Count' not in bias_df.columns:
        return None
    sub = bias_df[(bias_df['Architecture'] == 'HyperloopBERT') &
                  (bias_df['Model_Size'] == 'base') &
                  (bias_df['Band'] == band) &
                  (bias_df['Stream_Count'] == n)]
    if 'Merge_At' in sub.columns:
        sub = sub[sub['Merge_At'].isna()]
    if sub.empty:
        return None
    return float(sub['Overall_Stereotype_Preference_Rate'].mean())


def looped_rate(bias_df, band):
    sub = bias_df[(bias_df['Architecture'] == 'LoopedBERT') &
                  (bias_df['Model_Size'] == 'base') &
                  (bias_df['Band'] == band)]
    if sub.empty:
        return None
    return float(sub['Overall_Stereotype_Preference_Rate'].mean())


def signed_disagreement_correlation(disagreement_df):
    """Signed Pearson r between stream disagreement and bias effect size.
    The mechanistic prediction is NEGATIVE (more disagreement -> less bias)."""
    if disagreement_df is None or disagreement_df.empty or 'Pearson_R' not in disagreement_df.columns:
        return None
    r = disagreement_df['Pearson_R'].dropna()
    return float(r.mean()) if not r.empty else None


def decision_logic(confirmatory, bias_df, band, disagreement_df):
    """
    Stage 3 verdict per the pre-registered plan. The Stage 1-2 finding stands
    regardless; this classifies the strength of the Hyperloop-specific claim.
    Verdicts: GO / INVESTIGATE / PUBLISH-NULL.
    """
    by_name = {c['Contrast']: c for c in confirmatory}
    v_vs_h = by_name.get('VanillaBERT_vs_HyperloopBERT')
    l_vs_h = by_name.get('LoopedBERT_vs_HyperloopBERT')
    n4n1 = by_name.get('Hyperloop_n4_vs_n1')

    if v_vs_h is None:
        return ("INVESTIGATE (primary contrast Vanilla vs Hyperloop not computable "
                "- check training/eval coverage before interpreting anything)")

    primary_holds = v_vs_h['Mean_Delta'] > 0 and bool(v_vs_h.get('Significant'))
    # Spec allows Hyperloop <= Looped: Hyperloop must not be WORSE than Looped
    hyperloop_not_worse = l_vs_h is None or l_vs_h['Mean_Delta'] > -MONOTONE_TOLERANCE
    hyperloop_better = (l_vs_h is not None and l_vs_h['Mean_Delta'] > MONOTONE_TOLERANCE
                        and bool(l_vs_h.get('Significant')))
    dose_response = (n4n1 is not None and n4n1['Mean_Delta'] < -MONOTONE_TOLERANCE
                     and bool(n4n1.get('Significant')))

    # Monotonicity n1 >= n2 >= n4 in bias (within tolerance)
    r1, r2, r4 = (stream_rate(bias_df, band, n) for n in (1, 2, 4))
    monotone = None
    if None not in (r1, r2, r4):
        monotone = (r1 >= r2 - MONOTONE_TOLERANCE) and (r2 >= r4 - MONOTONE_TOLERANCE)

    # n=1 ~= Looped sanity (approximate: n=1 keeps its projections)
    rl = looped_rate(bias_df, band)
    sanity_ok = None
    if r1 is not None and rl is not None:
        sanity_ok = abs(r1 - rl) <= SANITY_TOLERANCE

    r_signed = signed_disagreement_correlation(disagreement_df)
    corr_supports = r_signed is not None and r_signed < -0.4  # SIGNED, corroborating only

    detail = (f"[primary_holds={primary_holds}, hyperloop_better={hyperloop_better}, "
              f"not_worse={hyperloop_not_worse}, dose_response={dose_response}, "
              f"monotone(n)={monotone} (rates n1/n2/n4={r1}/{r2}/{r4}), "
              f"n1~Looped={sanity_ok} (looped={rl}), signed_r={r_signed}]")
    logger.info(f"Stage 3 decision inputs: {detail}")

    if sanity_ok is False:
        return (f"INVESTIGATE (n=1 Hyperloop deviates from LoopedBERT beyond "
                f"tolerance {SANITY_TOLERANCE}: the stream implementation may "
                f"not collapse as intended; audit before any claim) {detail}")

    if not primary_holds:
        return (f"PUBLISH-NULL (primary contrast Vanilla vs Hyperloop not "
                f"significant after Holm; report per pre-registration) {detail}")

    if hyperloop_better and dose_response and (monotone is not False):
        strength = "with corroborating stream-disagreement correlation" if corr_supports \
                   else "mechanistic correlation weak/absent - frame as corroborating only"
        return f"GO (streams reduce bias: dose-response holds; {strength}) {detail}"

    if hyperloop_better and not dose_response:
        return (f"INVESTIGATE (Hyperloop < Looped but stream dose-response "
                f"inconclusive - streams framed as exploratory) {detail}")

    if hyperloop_not_worse:
        return (f"PUBLISH-NULL (Looped ~= Hyperloop within tolerance; null "
                f"Hyperloop result, publishable per pre-registration) {detail}")

    return f"INVESTIGATE (Hyperloop worse than Looped - unexpected; audit) {detail}"


def write_paper_outline(results_dir, confirmatory, exploratory, decision):
    outline_path = os.path.join(results_dir, 'stage3_paper_outline.md')
    is_go = decision.startswith('GO')

    if is_go:
        abstract = ("This work introduces HyperloopBERT, a compute-matched multi-stream "
                    "looped encoder with CLS-weighted stream aggregation. At matched "
                    "validation loss, stereotype preference decreases with stream count "
                    "in a from-scratch dose-response, extending the parameter-sharing "
                    "finding with a mechanistic account.")
    else:
        abstract = ("[Conditional: verdict was not GO. If PUBLISH-NULL, frame the "
                    "paper on the Stages 1-2 finding with Hyperloop as a tested-and-"
                    "bounded mechanistic extension.]")

    outline = f"""# Stage 3 Paper Outline

## Title
Parameter Sharing Reduces Stereotype Memorization: Mechanistic Evidence from
Looped and Hyper-Connected Encoders

## Finding-First Abstract
{abstract}

## Verdict
**{decision}**

### Confirmatory family (item-level primary, Holm corrected):
"""
    for c in confirmatory:
        outline += (f"- **{c['Contrast']}**: Delta = {c['Mean_Delta']:.4f} "
                    f"(p_raw = {c['P_Value']:.4f}, p_holm = {c.get('Holm_Corrected_P_Value')}, "
                    f"significant = {c.get('Significant')}, n_items = {c.get('N_Items')})\n")

    if exploratory:
        outline += "\n### Exploratory (uncorrected; corroborating only, never causal):\n"
        for c in exploratory:
            outline += f"- {c['Contrast']}: Delta = {c['Mean_Delta']:.4f} (p = {c['P_Value']:.4f})\n"

    outline += """
## Limitations
- Early-merge is an OOD intervention on a trained model: corroborating, not causal
- Stream-disagreement correlation is correlational; the causal claim rests only
  on the from-scratch stream dose-response
- Hyperloop carries extra hyper-connection parameters vs Looped (disclosed;
  the ablation direction is conservative with respect to this gap)
- n=1 collapse to Looped is approximate (projections remain)
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

    mlm_df, bias_df, indian_df, glue_df, disagreement_df, early_merge_df = load_stage3_data(results_dir)

    if mlm_df is None or bias_df is None:
        logger.error("Required data not found. Run evaluation first.")
        return

    primary_bands = {}
    for size in cfg.SIZES:
        band = compute_primary_band(mlm_df, cfg.ARCHITECTURES, size, logger_obj=logger)
        primary_bands[size] = band
        logger.info(f"Primary band for size {size}: {band}")
    band = primary_bands.get('base')

    metric = 'Overall_Stereotype_Preference_Rate'

    # Confirmatory family (item-level primary)
    confirmatory = []
    pair_specs = [
        ('VanillaBERT', None, 'HyperloopBERT', 4, 'greater'),
        ('VanillaBERT', None, 'LoopedBERT', None, 'greater'),
        ('LoopedBERT', None, 'HyperloopBERT', 4, 'greater'),
    ]
    if band is not None:
        for arch1, sc1, arch2, sc2, alt in pair_specs:
            item = compute_item_level_contrast(results_dir, 'multicrows', arch1, arch2,
                                               'base', band, stream_count1=sc1,
                                               stream_count2=sc2, alternative=alt)
            if item is None:
                logger.warning(f"{arch1}_vs_{arch2}: item-level contrast not computable.")
                continue
            confirmatory.append({
                'Contrast': f"{arch1}_vs_{arch2}",
                'Metric': 'Effect_Size (item-level)',
                'Dataset': 'Multi-CrowS-Pairs', 'Band': band, 'Model_Size': 'base',
                'Mean_Delta': item['mean_delta'], 'P_Value': item['p_value'],
                'CI_Low': item['ci_low'], 'CI_High': item['ci_high'],
                'Cohens_D': item['cohens_d'], 'N_Items': item['n_items'],
                'Preference_Rate_Delta': item['preference_rate_delta'],
            })
        # n=4 vs n=1 (expected NEGATIVE delta)
        n41 = compute_item_level_contrast(results_dir, 'multicrows', 'HyperloopBERT',
                                          'HyperloopBERT', 'base', band,
                                          stream_count1=4, stream_count2=1,
                                          alternative='less')
        if n41 is not None:
            confirmatory.append({
                'Contrast': 'Hyperloop_n4_vs_n1',
                'Metric': 'Effect_Size (item-level)',
                'Dataset': 'Multi-CrowS-Pairs', 'Band': band, 'Model_Size': 'base',
                'Mean_Delta': n41['mean_delta'], 'P_Value': n41['p_value'],
                'CI_Low': n41['ci_low'], 'CI_High': n41['ci_high'],
                'Cohens_D': n41['cohens_d'], 'N_Items': n41['n_items'],
                'Preference_Rate_Delta': n41['preference_rate_delta'],
            })

    p_values = [(c['Contrast'], c['P_Value']) for c in confirmatory]
    corrected = holm_bonferroni_correction(p_values)
    cmap = {name: (corr, sig) for name, _, corr, sig in corrected}
    for c in confirmatory:
        corr, sig = cmap.get(c['Contrast'], (None, False))
        c['Holm_Corrected_P_Value'] = corr
        c['Significant'] = sig

    # Exploratory: n=2 arm and early-merge (uncorrected, two-sided)
    exploratory = []
    if band is not None:
        n42 = compute_item_level_contrast(results_dir, 'multicrows', 'HyperloopBERT',
                                          'HyperloopBERT', 'base', band,
                                          stream_count1=4, stream_count2=2,
                                          alternative='two-sided')
        if n42 is not None:
            exploratory.append({'Contrast': 'Hyperloop_n4_vs_n2', 'Band': band,
                                'Mean_Delta': n42['mean_delta'], 'P_Value': n42['p_value'],
                                'N_Items': n42['n_items'],
                                'Note': 'EXPLORATORY: dose-response interior point'})

    # Decision
    decision = decision_logic(confirmatory, bias_df, band, disagreement_df)
    logger.info(f"=== STAGE 3 VERDICT: {decision} ===")

    # Save stats with the REGISTERED filenames
    stats_dir = os.path.join(results_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    if confirmatory:
        pd.DataFrame(confirmatory).to_csv(os.path.join(stats_dir, 'confirmatory_family.csv'), index=False)
    if exploratory:
        pd.DataFrame(exploratory).to_csv(os.path.join(stats_dir, 'exploratory_results.csv'), index=False)

    # Outline
    write_paper_outline(results_dir, confirmatory, exploratory, decision)

    # Figures
    ablation_df = None
    if 'Stream_Count' in bias_df.columns:
        ablation_df = bias_df[(bias_df['Architecture'] == 'HyperloopBERT') &
                              (bias_df['Stream_Count'].notna())].copy()
    if ablation_df is not None and not ablation_df.empty and band is not None:
        plot_stream_ablation(ablation_df, metric, band,
                             os.path.join(figures_dir, 'stream_ablation.png'))
    if disagreement_df is not None and not disagreement_df.empty:
        plot_stream_disagreement(disagreement_df,
                                 os.path.join(figures_dir, 'stream_disagreement.png'))
    plot_iso_loss_bias(bias_df, 'Multi-CrowS-Pairs', metric, 'base',
                       os.path.join(figures_dir, 'iso_loss_multicrows_base.png'))

    logger.info("Stage 3 Analysis complete.")


if __name__ == "__main__":
    main()

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
from common.stats_engine import (
    paired_permutation_test, cohens_d, bootstrap_ci, item_level_paired_contrast,
    binomial_p_above_chance,
)
from common.plotting import plot_iso_loss_bias, plot_token_budget_bias
import Stage1.config_stage1 as cfg

logger = setup_logging('analyze_stage1')

# PRE-REGISTERED PRIMARY TEST: item-level paired permutation (sentence pairs
# are the resampling unit) on per-item Effect_Size deltas at the primary
# iso-loss band, Multi-CrowS-Pairs, base size. Seed-level testing is retained
# as a robustness check only: with 3 seeds its minimum achievable p is
# 1/2^3 = 0.125, so it can NEVER clear 0.05 and must not gate decisions.


def load_data(results_dir):
    """Load summary tables."""
    mlm_path = os.path.join(results_dir, 'mlm', 'summary_table.csv')
    bias_path = os.path.join(results_dir, 'bias', 'multicrows_summary.csv')
    indian_path = os.path.join(results_dir, 'bias', 'indian_bias_summary.csv')

    mlm_df = pd.read_csv(mlm_path) if os.path.exists(mlm_path) else None
    bias_df = pd.read_csv(bias_path) if os.path.exists(bias_path) else None
    indian_df = pd.read_csv(indian_path) if os.path.exists(indian_path) else None

    return mlm_df, bias_df, indian_df


def _load_item_frame(results_dir, ds_prefix, arch, size, band, stream_count=None):
    """
    Load per-item rows for one (architecture, size, band) across all seeds
    from the per-example progress CSVs. Returns a DataFrame with
    Row_Index, Seed, Effect_Size, Stereotype_Preferred (failed rows dropped,
    drop count logged).
    """
    suffix = f"_streams{stream_count}" if stream_count is not None else ""
    pattern = os.path.join(results_dir, 'bias',
                           f"{ds_prefix}_{arch}_{size}_seed*{suffix}_band{band}_progress.csv")
    frames = []
    for path in globmod.glob(pattern):
        # Guard: without an explicit stream suffix, exclude ablation files
        if stream_count is None and '_streams' in os.path.basename(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            continue
        cols = [c for c in ('Row_Index', 'Seed', 'Effect_Size', 'Stereotype_Preferred') if c in df.columns]
        frames.append(df[cols])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    n_total = len(out)
    out = out.dropna(subset=['Effect_Size'])
    dropped = n_total - len(out)
    if dropped:
        logger.warning(f"{arch}/{size}/band{band}: {dropped} failed items excluded from item-level test.")
    # One row per (Seed, Row_Index)
    out = out.drop_duplicates(subset=['Seed', 'Row_Index'], keep='last')
    return out


def compute_item_level_contrast(results_dir, ds_prefix, arch1, arch2, size, band,
                                stream_count1=None, stream_count2=None,
                                alternative='greater'):
    """
    PRIMARY statistical test. Per item: average Effect_Size across seeds within
    each architecture, then delta = arch1 - arch2; sign-flip permutation over
    items (n ~ dataset size, real power at any seed count).
    """
    df1 = _load_item_frame(results_dir, ds_prefix, arch1, size, band, stream_count1)
    df2 = _load_item_frame(results_dir, ds_prefix, arch2, size, band, stream_count2)
    if df1.empty or df2.empty:
        return None

    m1 = df1.groupby('Row_Index')['Effect_Size'].mean()
    m2 = df2.groupby('Row_Index')['Effect_Size'].mean()
    common = m1.index.intersection(m2.index)
    if len(common) < 10:
        return None
    deltas = (m1.loc[common] - m2.loc[common]).tolist()

    result = item_level_paired_contrast(deltas, alternative=alternative)

    # Descriptive: preference-rate delta on the same items
    p1 = df1.groupby('Row_Index')['Stereotype_Preferred'].mean().loc[common].mean()
    p2 = df2.groupby('Row_Index')['Stereotype_Preferred'].mean().loc[common].mean()
    result['preference_rate_1'] = float(p1)
    result['preference_rate_2'] = float(p2)
    result['preference_rate_delta'] = float(p1 - p2)
    return result


def compute_contrast(bias_df, arch1, arch2, metric_col, primary_bands):
    """
    Seed-level contrast between arch1 and arch2 at the primary band per size.
    ROBUSTNESS CHECK ONLY (see header note). Also audits the iso-loss match:
    reports the mean/max per-seed |validation-loss gap| between the two
    architectures' snapshots and flags gaps above cfg.ISO_LOSS_TOLERANCE.
    """
    results = {}
    tolerance = getattr(cfg, 'ISO_LOSS_TOLERANCE', 0.05)

    for size, band in primary_bands.items():
        if band is None:
            continue

        df1 = bias_df[(bias_df['Architecture'] == arch1) & (bias_df['Model_Size'] == size) & (bias_df['Band'] == band)]
        df2 = bias_df[(bias_df['Architecture'] == arch2) & (bias_df['Model_Size'] == size) & (bias_df['Band'] == band)]

        # Exclude ablation rows so the contrast pairs one row per seed
        if 'Stream_Count' in bias_df.columns:
            df1 = df1[df1['Stream_Count'].isna() | (df1['Stream_Count'] == 4)]
            df2 = df2[df2['Stream_Count'].isna() | (df2['Stream_Count'] == 4)]
        if 'Merge_At' in bias_df.columns:
            df1 = df1[df1['Merge_At'].isna()]
            df2 = df2[df2['Merge_At'].isna()]

        df1 = df1.sort_values('Seed').drop_duplicates(subset=['Seed'], keep='last')
        df2 = df2.sort_values('Seed').drop_duplicates(subset=['Seed'], keep='last')

        common_seeds = set(df1['Seed']).intersection(set(df2['Seed']))
        if not common_seeds:
            continue
        df1 = df1[df1['Seed'].isin(common_seeds)]
        df2 = df2[df2['Seed'].isin(common_seeds)]

        vals1 = df1[metric_col].tolist()
        vals2 = df2[metric_col].tolist()

        diffs = [a - b for a, b in zip(vals1, vals2)]
        mean_delta = sum(diffs) / len(diffs)

        # Iso-loss match audit
        mean_gap, max_gap = None, None
        if 'Validation_Loss' in df1.columns and 'Validation_Loss' in df2.columns:
            merged = pd.merge(df1[['Seed', 'Validation_Loss']],
                              df2[['Seed', 'Validation_Loss']],
                              on='Seed', suffixes=('_1', '_2')).dropna()
            if not merged.empty:
                gaps = (merged['Validation_Loss_1'] - merged['Validation_Loss_2']).abs()
                mean_gap, max_gap = float(gaps.mean()), float(gaps.max())
                if max_gap > tolerance:
                    logger.warning(
                        f"ISO-LOSS MISMATCH {arch1} vs {arch2} ({size}, band {band}): "
                        f"max |loss gap| = {max_gap:.4f} > tolerance {tolerance}. "
                        f"Report this gap in the paper; consider a finer band.")

        p_val = None
        ci_low, ci_high = None, None
        d = None
        if len(diffs) >= 2:
            _, p_val = paired_permutation_test(vals1, vals2)
            _, ci_low, ci_high = bootstrap_ci(diffs)
            d = cohens_d(vals1, vals2)

        results[size] = {
            'Contrast': f"{arch1}_vs_{arch2}",
            'Metric': metric_col,
            'Model_Size': size,
            'Band': band,
            'Seeds': len(diffs),
            'Mean_Delta': mean_delta,
            'P_Value': p_val,
            'CI_Low': ci_low,
            'CI_High': ci_high,
            'Cohens_D': d,
            'Mean_Loss_Gap': mean_gap,
            'Max_Loss_Gap': max_gap,
        }

    return results


def _glue_capability_leg(results_dir, band, alpha):
    """
    Leg 1: SST-2 + RTE (pulled forward from Stage 2) must be above chance on
    VanillaBERT base at the primary band -- one-sided exact binomial per task,
    accuracy pooled over seeds. Produced by eval_capability_stage1.py.
    """
    glue_path = os.path.join(results_dir, 'glue', 'summary_table.csv')
    if not os.path.exists(glue_path):
        return None, "no Stage 1 GLUE results (run eval_capability_stage1.py)"
    df = pd.read_csv(glue_path)
    df = df[(df['Architecture'] == 'VanillaBERT') & (df['Model_Size'] == 'base')
            & (df['Band'] == band)]
    if df.empty:
        return None, f"no VanillaBERT base GLUE rows at band {band}"

    details = []
    all_pass = True
    for task in cfg.GLUE_TASKS:
        sub = df[df['Task'] == task].drop_duplicates(subset=['Seed'], keep='last')
        if sub.empty:
            return None, f"GLUE task {task} missing at band {band}"
        n = int(sub['Eval_Example_Count'].fillna(0).sum())
        correct = int(round((sub['Accuracy'] * sub['Eval_Example_Count'].fillna(0)).sum()))
        if n == 0:
            return None, f"GLUE task {task}: no eval example counts recorded"
        p = binomial_p_above_chance(correct, n)
        ok = p < alpha
        all_pass = all_pass and ok
        details.append(f"{task} {correct}/{n} ({correct / n:.3f}, p={p:.2e}, "
                       f"{'PASS' if ok else 'FAIL'})")
    return all_pass, "; ".join(details)


def _coreference_capability_leg(results_dir, band, alpha):
    """
    Leg 3: WinoBias masked-pronoun accuracy on the PRO-stereotypical splits
    (pooled type1+type2) must be above chance on VanillaBERT base at the
    primary band. On pro items, stereotype knowledge and coreference ability
    point the same way, so at-chance accuracy here means no gendered pronoun
    signal was learned and Stage 2 WinoBias results would be noise.
    """
    path = os.path.join(results_dir, 'bias', 'winobias_capability.csv')
    if not os.path.exists(path):
        return None, "no WinoBias capability results (run eval_capability_stage1.py)"
    df = pd.read_csv(path)
    df = df[(df['Architecture'] == 'VanillaBERT') & (df['Model_Size'] == 'base')
            & (df['Band'] == band)]
    pro = df[df['Split'].isin(['type1_pro', 'type2_pro'])]
    pro = pro.drop_duplicates(subset=['Seed', 'Split'], keep='last')
    if pro.empty:
        return None, f"no pro-stereotypical WinoBias rows at band {band}"
    correct = int(pro['Correct_Count'].sum())
    n = int(pro['Scored_Count'].sum())
    if n == 0:
        return None, "WinoBias pro splits scored 0 items"
    p = binomial_p_above_chance(correct, n)
    ok = p < alpha
    return ok, (f"pro-split masked-pronoun {correct}/{n} "
                f"({correct / n:.3f}, p={p:.2e}, {'PASS' if ok else 'FAIL'})")


def capability_gate(results_dir, primary_bands):
    """
    Capability gate (undertrained-model guard), THREE pre-registered legs on
    VanillaBERT base at the primary band:
      Leg 1: GLUE screen (SST-2 + RTE above chance, exact binomial).
      Leg 2: baseline bias detectable (item-level preference-rate bootstrap
             CI excludes 0.5 from above). A model that never acquired the
             stereotypes cannot show an architecture 'reduces' them.
      Leg 3: WinoBias masked-pronoun pro-split accuracy above chance.
    Status: PASS (all three pass), FAIL (any leg fails), INCOMPLETE (a leg's
    evidence is missing -- contrasts must NOT be interpreted until it exists).
    """
    band = primary_bands.get('base')
    if band is None:
        return {'status': 'UNKNOWN', 'detail': 'no primary band at base size'}
    alpha = getattr(cfg, 'CAPABILITY_ALPHA', 0.05)

    # Leg 2 (baseline bias) -- unchanged primary leg
    df = _load_item_frame(results_dir, 'multicrows', 'VanillaBERT', 'base', band)
    if df.empty:
        return {'status': 'UNKNOWN', 'detail': 'no VanillaBERT base items found'}
    per_item = df.groupby('Row_Index')['Stereotype_Preferred'].mean().tolist()
    mean_rate, ci_low, ci_high = bootstrap_ci(per_item)
    leg2_pass = ci_low > 0.5
    leg2_detail = (f"Vanilla base preference {mean_rate:.3f} "
                   f"[{ci_low:.3f}, {ci_high:.3f}] vs chance 0.5 "
                   f"({'PASS' if leg2_pass else 'FAIL'})")

    leg1_pass, leg1_detail = _glue_capability_leg(results_dir, band, alpha)
    leg3_pass, leg3_detail = _coreference_capability_leg(results_dir, band, alpha)

    legs = {'leg1_glue': (leg1_pass, leg1_detail),
            'leg2_baseline_bias': (leg2_pass, leg2_detail),
            'leg3_coreference': (leg3_pass, leg3_detail)}
    detail = "; ".join(f"{name}: {d}" for name, (_, d) in legs.items())

    if any(ok is False for ok, _ in legs.values()):
        status = 'FAIL'
    elif any(ok is None for ok, _ in legs.values()):
        status = 'INCOMPLETE'
    else:
        status = 'PASS'
    return {
        'status': status,
        'baseline_preference_rate': mean_rate,
        'ci_low': ci_low, 'ci_high': ci_high,
        'legs': {name: ok for name, (ok, _) in legs.items()},
        'detail': detail,
    }


def go_nogo_decision(primary_contrast_results, item_result, capability, sizes, n_seeds):
    """
    Pre-registered Stage 1 decision:
      - CAPABILITY-FAIL if baseline bias is not detectable (gate 2.1).
      - NO-GO on directional reversal (delta < -0.02 at any size).
      - NO-GO if all |delta| < 0.02 (effect absent).
      - GO requires delta > 0.02 in >= 2 of 3 sizes AND the item-level primary
        test significant (p < 0.05, one-sided).
      - EXTEND-SEEDS only when directionally consistent but the primary test
        is not yet significant AND fewer than 3 seeds have been run. After 3
        seeds without significance the verdict is NO-GO, not endless extension.
    """
    if capability['status'] == 'FAIL':
        return ("CAPABILITY-FAIL (one or more capability legs failed on "
                "VanillaBERT base: the models are too undertrained for the bias "
                "instruments to measure anything. Do NOT interpret contrasts; "
                "increase token budget or corpus bias density first. "
                f"Legs: {capability.get('legs')})")
    if capability['status'] == 'INCOMPLETE':
        return ("CAPABILITY-INCOMPLETE (a capability leg's evidence is missing "
                "-- run Stage1/eval_capability_stage1.py, then re-run this "
                "analysis. Contrasts must NOT be interpreted before the gate "
                f"is complete. Legs: {capability.get('legs')})")

    deltas = {size: primary_contrast_results[size]['Mean_Delta']
              for size in sizes if size in primary_contrast_results}
    if not deltas:
        return "NO-GO (No data at primary bands)"

    num_sizes_positive = sum(1 for d in deltas.values() if d > 0.02)
    has_reversal = any(d < -0.02 for d in deltas.values())
    all_null = all(abs(d) < 0.02 for d in deltas.values())

    item_sig = item_result is not None and item_result['p_value'] < 0.05 and item_result['mean_delta'] > 0

    if has_reversal:
        return "NO-GO (directional reversal at one or more sizes)"
    if all_null:
        return "NO-GO (mean |delta| < 0.02 at every size: effect absent)"
    if num_sizes_positive >= 2 and item_sig:
        return "GO"
    if num_sizes_positive >= 2 and not item_sig and n_seeds < 3:
        return "EXTEND-SEEDS (directionally consistent; primary item-level test not yet significant)"
    return "NO-GO (direction inconsistent or primary test not significant after 3 seeds)"


def secondary_indian_report(results_dir, primary_bands):
    """Secondary confirmation: direction of the Vanilla-Looped delta on the
    Indian bias instrument (reported, never gated on)."""
    band = primary_bands.get('base')
    if band is None:
        return None
    res = compute_item_level_contrast(results_dir, 'indian_bias', 'VanillaBERT',
                                      'LoopedBERT', 'base', band)
    return res


def write_paper_outline(results_dir, primary_contrast_results, item_result,
                        indian_result, capability, decision):
    """Write stage1_paper_outline.md. The finding is asserted ONLY on GO."""

    outline_path = os.path.join(results_dir, 'stage1_paper_outline.md')
    is_go = decision.startswith('GO')

    if is_go and item_result is not None:
        abstract = (
            f"At matched validation loss (iso-loss), looped transformers with "
            f"cross-layer parameter sharing show a "
            f"{item_result['preference_rate_delta'] * 100:.1f}-point lower stereotype "
            f"preference rate than vanilla transformers on Multi-CrowS-Pairs English "
            f"(item-level paired permutation, p = {item_result['p_value']:.4f}, "
            f"n = {item_result['n_items']} pairs), consistent across "
            f"{sum(1 for r in primary_contrast_results.values() if r['Mean_Delta'] > 0.02)} "
            f"of 3 model scales.")
    else:
        abstract = ("[NOT ASSERTED -- verdict was not GO. Do not write a finding "
                    "abstract from this run.]")

    outline = f"""# Stage 1 Paper Outline

## Title
Parameter Sharing Reduces Stereotype Memorization: A Controlled Study Across Model Scale

## Finding-First Abstract
{abstract}

## Capability Gate
{capability.get('detail', 'unknown')} -> {capability['status']}

## Method
- Two architectures (VanillaBERT, LoopedBERT), compute-matched at effective depth 12
- Three model scales (hidden=256/512/768)
- Primary comparison: matched validation loss (iso-loss protocol; per-contrast
  loss gaps audited against tolerance {getattr(cfg, 'ISO_LOSS_TOLERANCE', 0.05)})
- Primary test: item-level paired permutation on per-pair Effect_Size deltas
  (sentence pairs as resampling unit); seed-level contrast reported as robustness
- Primary instrument: Multi-CrowS-Pairs English; secondary: Indian bias instrument

## Verdict
**{decision}**

### Primary item-level contrast (base size):
"""
    if item_result is not None:
        outline += (f"- Delta(Effect_Size) = {item_result['mean_delta']:.4f} "
                    f"[{item_result['ci_low']:.4f}, {item_result['ci_high']:.4f}], "
                    f"p = {item_result['p_value']:.4f}, d = {item_result['cohens_d']:.3f}, "
                    f"n = {item_result['n_items']}\n"
                    f"- Preference rates: Vanilla {item_result['preference_rate_1']:.3f} "
                    f"vs Looped {item_result['preference_rate_2']:.3f}\n")
    else:
        outline += "- Not computable (missing per-item data)\n"

    outline += "\n### Seed-level contrasts (robustness only):\n"
    for size, res in primary_contrast_results.items():
        outline += (f"- **{size.capitalize()}**: Delta = {res['Mean_Delta']:.3f} "
                    f"(seeds = {res['Seeds']}, loss gap mean/max = "
                    f"{res['Mean_Loss_Gap']}/{res['Max_Loss_Gap']})\n")

    if indian_result is not None:
        outline += (f"\n### Secondary (Indian instrument, base size):\n"
                    f"- Delta(Effect_Size) = {indian_result['mean_delta']:.4f}, "
                    f"p = {indian_result['p_value']:.4f} -- direction "
                    f"{'AGREES' if indian_result['mean_delta'] > 0 else 'DISAGREES'} "
                    f"with the primary instrument\n")

    outline += """
## Construct scope (state this BEFORE the limitations in the paper)
- The primary endpoint is a CORRELATION-type fairness measure in the
  descriptive / normative / correlation taxonomy of Wang et al. (2025,
  "Fairness through Difference Awareness", ACL 2025 Best Paper,
  arXiv:2502.01926). We measure stereotype ASSOCIATION at matched model
  quality; we do NOT claim difference-aware fairness.

## Limitations
- Scratch pretraining at 200M tokens; not SOTA quality
- PLL construct-validity caveat (Blodgett et al. 2021); strengthened in Stage 2 with SS-PLL
- English-only primary analysis
- Iso-loss matching is banded, not exact; per-contrast loss gaps reported above
- A lower stereotype preference rate is not automatically desirable. Wang et al.
  (2025) show difference-UNAWARE treatment is not universally the correct target
  and that bias-mitigation strategies can backfire on difference-aware tasks; in
  principle the same mechanism that suppresses stereotype association could also
  erode LEGITIMATE group distinctions.

## Future work (stated failure mode of SCH)
- SCH predicts weight sharing reduces TOTAL stereotype storage at matched
  quality. If that mechanism is non-selective, it should ALSO degrade
  legitimate DESCRIPTIVE group knowledge. Testing that needs
  instruction-tuned-scale models and a difference-awareness benchmark
  (Wang et al. 2025); it is out of reach at this token budget, where every
  arm would sit at chance on such a benchmark.
"""

    os.makedirs(os.path.dirname(outline_path), exist_ok=True)
    with open(outline_path, 'w') as f:
        f.write(outline)

    logger.info(f"Paper outline written to {outline_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-namespace', type=str, default=cfg.VAL_DATASET_NAMESPACE)
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds that were trained (for the decision rule)')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(base_dir, cfg.RESULTS_DIR)
    figures_dir = os.path.join(base_dir, cfg.FIGURES_DIR)

    mlm_df, bias_df, indian_df = load_data(results_dir)

    if mlm_df is None or bias_df is None:
        logger.error("Required data not found. Run evaluation first.")
        return

    # 1. Primary bands (per-seed coverage enforced)
    primary_bands = {}
    for size in cfg.SIZES:
        band = compute_primary_band(mlm_df, cfg.ARCHITECTURES, size, logger_obj=logger)
        primary_bands[size] = band
        logger.info(f"Primary band for size {size}: {band}")

    # 2. Capability gate
    capability = capability_gate(results_dir, primary_bands)
    logger.info(f"Capability gate: {capability['status']} -- {capability.get('detail')}")

    # 3. Primary item-level contrast (base size)
    item_result = None
    if primary_bands.get('base') is not None:
        item_result = compute_item_level_contrast(
            results_dir, 'multicrows', 'VanillaBERT', 'LoopedBERT',
            'base', primary_bands['base'])
        if item_result:
            logger.info(f"PRIMARY item-level contrast: delta={item_result['mean_delta']:.4f}, "
                        f"p={item_result['p_value']:.4f}, n={item_result['n_items']}")

    # 4. Seed-level robustness contrasts
    metric = 'Overall_Stereotype_Preference_Rate'
    primary_contrast_results = compute_contrast(bias_df, 'VanillaBERT', 'LoopedBERT', metric, primary_bands)

    # 5. Secondary: Indian instrument direction
    indian_result = secondary_indian_report(results_dir, primary_bands)

    # 6. Decision
    decision = go_nogo_decision(primary_contrast_results, item_result, capability,
                                cfg.SIZES, args.seeds)
    logger.info(f"=== GO/NO-GO VERDICT: {decision} ===")

    # 7. Save contrast CSV
    contrast_rows = []
    for res in primary_contrast_results.values():
        contrast_rows.append({
            'Contrast': res['Contrast'],
            'Metric': res['Metric'],
            'Dataset': 'Multi-CrowS-Pairs',
            'Band': res['Band'],
            'Model_Size': res['Model_Size'],
            'Mean_Delta_Preference': res['Mean_Delta'],
            'Bootstrap_CI_Low': res['CI_Low'],
            'Bootstrap_CI_High': res['CI_High'],
            'Permutation_P_Value': res['P_Value'],
            'Cohens_D': res['Cohens_D'],
            'Timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    if item_result is not None:
        contrast_rows.append({
            'Contrast': 'VanillaBERT_vs_LoopedBERT_ITEM_LEVEL_PRIMARY',
            'Metric': 'Effect_Size',
            'Dataset': 'Multi-CrowS-Pairs',
            'Band': primary_bands.get('base'),
            'Model_Size': 'base',
            'Mean_Delta_Preference': item_result['mean_delta'],
            'Bootstrap_CI_Low': item_result['ci_low'],
            'Bootstrap_CI_High': item_result['ci_high'],
            'Permutation_P_Value': item_result['p_value'],
            'Cohens_D': item_result['cohens_d'],
            'Timestamp': datetime.utcnow().isoformat() + 'Z'
        })

    if contrast_rows:
        stats_dir = os.path.join(results_dir, 'stats')
        os.makedirs(stats_dir, exist_ok=True)
        pd.DataFrame(contrast_rows).to_csv(os.path.join(stats_dir, 'primary_contrast.csv'), index=False)

    # 8. Outline (finding asserted only on GO)
    write_paper_outline(results_dir, primary_contrast_results, item_result,
                        indian_result, capability, decision)

    # 9. Figures
    for size in cfg.SIZES:
        plot_iso_loss_bias(bias_df, 'Multi-CrowS-Pairs', metric, size,
                           os.path.join(figures_dir, f'iso_loss_multicrows_{size}.png'))
        plot_token_budget_bias(bias_df, 'Multi-CrowS-Pairs', metric, size,
                               os.path.join(figures_dir, f'token_budget_multicrows_{size}.png'))
        if indian_df is not None:
            plot_iso_loss_bias(indian_df, 'Indian Bias (English)', metric, size,
                               os.path.join(figures_dir, f'iso_loss_indian_{size}.png'))

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()

import os
import sys
import pandas as pd
import argparse
from datetime import datetime

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.iso_loss import compute_primary_band
from common.stats_engine import paired_permutation_test, cohens_d, bootstrap_ci
from common.plotting import plot_iso_loss_bias, plot_token_budget_bias
import Stage1.config_stage1 as cfg

logger = setup_logging('analyze_stage1')

def load_data(results_dir):
    """Load summary tables."""
    mlm_path = os.path.join(results_dir, 'mlm', 'summary_table.csv')
    bias_path = os.path.join(results_dir, 'bias', 'multicrows_summary.csv')
    indian_path = os.path.join(results_dir, 'bias', 'indian_bias_summary.csv')
    
    mlm_df = pd.read_csv(mlm_path) if os.path.exists(mlm_path) else None
    bias_df = pd.read_csv(bias_path) if os.path.exists(bias_path) else None
    indian_df = pd.read_csv(indian_path) if os.path.exists(indian_path) else None
    
    return mlm_df, bias_df, indian_df

def compute_contrast(bias_df, arch1, arch2, metric_col, primary_bands):
    """
    Compute contrast between arch1 and arch2 at the primary band for each size.
    Returns delta (arch1 - arch2), p_value, ci_low, ci_high.
    """
    results = {}
    
    for size, band in primary_bands.items():
        if band is None:
            continue
            
        df1 = bias_df[(bias_df['Architecture'] == arch1) & (bias_df['Model_Size'] == size) & (bias_df['Band'] == band)]
        df2 = bias_df[(bias_df['Architecture'] == arch2) & (bias_df['Model_Size'] == size) & (bias_df['Band'] == band)]
        
        # Sort by seed to pair them correctly
        df1 = df1.sort_values('Seed')
        df2 = df2.sort_values('Seed')
        
        # Ensure we have matching seeds
        common_seeds = set(df1['Seed']).intersection(set(df2['Seed']))
        if not common_seeds:
            continue
            
        df1 = df1[df1['Seed'].isin(common_seeds)]
        df2 = df2[df2['Seed'].isin(common_seeds)]
        
        vals1 = df1[metric_col].tolist()
        vals2 = df2[metric_col].tolist()
        
        diffs = [a - b for a, b in zip(vals1, vals2)]
        mean_delta = sum(diffs) / len(diffs)
        
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
            'Cohens_D': d
        }
        
    return results

def go_nogo_decision(primary_contrast_results, sizes):
    """Evaluate go/no-go logic based on predefined criteria."""
    
    # Extract deltas
    deltas = {}
    for size in sizes:
        if size in primary_contrast_results:
            deltas[size] = primary_contrast_results[size]['Mean_Delta']
            
    if not deltas:
        return "NO-GO (No data at primary bands)"
        
    num_sizes_positive = sum(1 for d in deltas.values() if d > 0.02)
    has_reversal = any(d < -0.02 for d in deltas.values())
    
    # Check significance
    sig_count = 0
    for res in primary_contrast_results.values():
        if res['P_Value'] is not None and res['P_Value'] < 0.05:
            sig_count += 1
        elif res['CI_Low'] is not None and res['CI_Low'] > 0:
            sig_count += 1
            
    has_sig = sig_count > 0
    
    if num_sizes_positive >= 2 and not has_reversal and has_sig:
        return "GO"
    elif num_sizes_positive >= 2 and not has_reversal and not has_sig:
        return "EXTEND-SEEDS (Directionally correct, need more statistical power)"
    else:
        return "NO-GO"

def write_paper_outline(results_dir, primary_contrast_results, decision):
    """Write the stage1_paper_outline.md"""
    
    outline_path = os.path.join(results_dir, 'stage1_paper_outline.md')
    
    # Extract base size result for the abstract if available
    base_delta = "X"
    if 'base' in primary_contrast_results:
        base_delta = f"{primary_contrast_results['base']['Mean_Delta'] * 100:.1f}"
        
    num_sizes = len(primary_contrast_results)
        
    outline = f"""# Stage 1 Paper Outline

## Title
Parameter Sharing Reduces Stereotype Memorization: A Controlled Study Across Model Scale

## Finding-First Abstract
At matched validation loss (iso-perplexity), looped transformers with cross-layer parameter
sharing show {base_delta}% lower stereotype preference than vanilla transformers,
consistently across {num_sizes} of 3 model scales.

## Method
- Two architectures (VanillaBERT, LoopedBERT), compute-matched at effective depth 12
- Three model scales (hidden=256/512/768)
- Primary comparison: matched validation loss (iso-loss protocol)
- Primary metric: Multi-CrowS-Pairs English PLL (9 categories)
- Geographic novelty: Indian Multilingual Bias Dataset (Caste/Gender/Religion/Race)

## Go/No-Go Statement
**Verdict: {decision}**

### Primary Contrast Details:
"""
    for size, res in primary_contrast_results.items():
        outline += f"- **{size.capitalize()}**: Delta = {res['Mean_Delta']:.3f} "
        if res['P_Value'] is not None:
            outline += f"(p={res['P_Value']:.3f}, CI: [{res['CI_Low']:.3f}, {res['CI_High']:.3f}])"
        outline += "\n"
        
    outline += """
## Limitations
- Scratch pretraining at 200M tokens; not SOTA quality
- PLL construct-validity caveat (Blodgett et al. 2021); to be strengthened in Stage 2 with SS-PLL
- English-only primary analysis
- Single-GPU sequential training
"""

    os.makedirs(os.path.dirname(outline_path), exist_ok=True)
    with open(outline_path, 'w') as f:
        f.write(outline)
        
    logger.info(f"Paper outline written to {outline_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-namespace', type=str, default=cfg.VAL_DATASET_NAMESPACE)
    args = parser.parse_args()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(base_dir, cfg.RESULTS_DIR)
    figures_dir = os.path.join(base_dir, cfg.FIGURES_DIR)
    
    mlm_df, bias_df, indian_df = load_data(results_dir)
    
    if mlm_df is None or bias_df is None:
        logger.error("Required data not found. Run evaluation first.")
        return
        
    # 1. Determine primary bands
    primary_bands = {}
    for size in cfg.SIZES:
        band = compute_primary_band(mlm_df, cfg.ARCHITECTURES, size)
        primary_bands[size] = band
        logger.info(f"Primary band for size {size}: {band}")
        
    # 2. Compute Contrast (Vanilla vs Looped on Overall_Stereotype_Preference_Rate)
    metric = 'Overall_Stereotype_Preference_Rate'
    primary_contrast_results = compute_contrast(bias_df, 'VanillaBERT', 'LoopedBERT', metric, primary_bands)
    
    # 3. Decision
    decision = go_nogo_decision(primary_contrast_results, cfg.SIZES)
    logger.info(f"=== GO/NO-GO VERDICT: {decision} ===")
    
    for size, res in primary_contrast_results.items():
        logger.info(f"Size {size}: Delta={res['Mean_Delta']:.3f}, p={res['P_Value']}")
        
    # 4. Save Contrast CSV
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
        
    if contrast_rows:
        stats_dir = os.path.join(results_dir, 'stats')
        os.makedirs(stats_dir, exist_ok=True)
        pd.DataFrame(contrast_rows).to_csv(os.path.join(stats_dir, 'primary_contrast.csv'), index=False)
        
    # 5. Outline
    write_paper_outline(results_dir, primary_contrast_results, decision)
    
    # 6. Figures
    for size in cfg.SIZES:
        # Headline Plot
        plot_iso_loss_bias(bias_df, 'Multi-CrowS-Pairs', metric, size, 
                           os.path.join(figures_dir, f'iso_loss_multicrows_{size}.png'))
                           
        # Secondary Plot
        plot_token_budget_bias(bias_df, 'Multi-CrowS-Pairs', metric, size, 
                               os.path.join(figures_dir, f'token_budget_multicrows_{size}.png'))

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()

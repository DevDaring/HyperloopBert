import os
import sys
import pandas as pd
import argparse
from datetime import datetime

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.iso_loss import compute_primary_band
from common.stats_engine import paired_permutation_test, cohens_d, bootstrap_ci, exploratory_contrast
from common.plotting import plot_stream_ablation, plot_stream_disagreement
from Stage1.analyze_stage1 import load_data, compute_contrast
import Stage3.config_stage3 as cfg

logger = setup_logging('analyze_stage3')

def load_stage3_data(results_dir):
    mlm_df, bias_df, indian_df = load_data(results_dir)
    
    ablation_path = os.path.join(results_dir, 'bias', 'multicrows_ablation_summary.csv')
    glue_path = os.path.join(results_dir, 'glue', 'summary_table.csv')
    disagreement_path = os.path.join(results_dir, 'mechanistic', 'stream_disagreement.csv')
    
    ablation_df = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else None
    glue_df = pd.read_csv(glue_path) if os.path.exists(glue_path) else None
    disagreement_df = pd.read_csv(disagreement_path) if os.path.exists(disagreement_path) else None
    
    return mlm_df, bias_df, indian_df, ablation_df, glue_df, disagreement_df

def decision_logic(contrasts, disagreement_df):
    """
    GO if: HyperloopBERT outperforms EarlyMergeHyperloopBERT AND stream correlation > 0.4
    PAUSE if: HyperloopBERT outperforms but correlation weak, or vice versa
    NO-GO if: EarlyMerge outperforms HyperloopBERT or no difference
    """
    h_vs_e = next((c for c in contrasts if c['Contrast'] == 'HyperloopBERT_vs_EarlyMergeHyperloopBERT'), None)
    
    if not h_vs_e:
        return "NO-GO (Missing data)"
        
    # We want EarlyMerge to have HIGHER bias than Hyperloop, 
    # meaning Hyperloop reduces bias.
    # So Delta (Hyperloop - EarlyMerge) should be negative, or 
    # if using generic compute_contrast which does A - B:
    # A=Hyperloop, B=EarlyMerge -> Mean_Delta < -0.02
    
    h_better = h_vs_e['Mean_Delta'] < -0.02
    
    corr_strong = False
    if disagreement_df is not None and not disagreement_df.empty:
        # Check Pearson/Spearman
        if 'Pearson_R' in disagreement_df.columns:
            r = disagreement_df['Pearson_R'].mean()
            if pd.notna(r) and abs(r) > 0.4:
                corr_strong = True
                
    if h_better and corr_strong:
        return "GO"
        
    if h_better and not corr_strong:
        return "PAUSE (Hyperloop better but stream correlation weak)"
        
    if not h_better:
        return "NO-GO (EarlyMerge outperforms or no difference)"
        
    return "PAUSE"

def write_paper_outline(results_dir, contrasts, decision, disagreement_df):
    outline_path = os.path.join(results_dir, 'stage3_paper_outline.md')
    
    outline = f"""# Stage 3 Paper Outline

## Title
Hyperloop Transformers: Routing Computations to Mitigate Stereotype Memorization

## Finding-First Abstract
We introduce HyperloopBERT, an architecture that uses parallel computational streams within a shared
Transformer block. We show that maintaining independent streams until late-stage aggregation (CWSA) 
significantly reduces stereotype memorization compared to early-merging baselines, with stream 
disagreement strongly predicting bias mitigation.

## Method
- Novel Architecture: HyperloopBERT (depth/width routing, CWSA)
- Control: EarlyMergeHyperloopBERT
- Ablation: 1, 2, 4, 8 streams
- Mechanistic: Stream disagreement correlation

## Go/No-Go Statement
**Verdict: {decision}**

### Key Findings:
"""
    for c in contrasts:
        outline += f"- **{c['Contrast']}**: Delta = {c['Mean_Delta']:.3f} "
        if c.get('P_Value') is not None:
            outline += f"(p={c['P_Value']:.3f})"
        outline += "\n"
        
    if disagreement_df is not None and not disagreement_df.empty:
        r = disagreement_df['Pearson_R'].mean()
        p = disagreement_df['Pearson_P'].mean()
        outline += f"- **Stream Disagreement**: Pearson r = {r:.3f} (p={p:.3f})\n"

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
    
    mlm_df, bias_df, indian_df, ablation_df, glue_df, disagreement_df = load_stage3_data(results_dir)
    
    if mlm_df is None or bias_df is None:
        logger.error("Required data not found. Run evaluation first.")
        return
        
    primary_bands = {}
    for size in cfg.SIZES:
        band = compute_primary_band(mlm_df, cfg.ARCHITECTURES, size)
        primary_bands[size] = band
        logger.info(f"Primary band for size {size}: {band}")
        
    metric = 'Overall_Stereotype_Preference_Rate'
    
    # 1. Exploratory Contrast: Hyperloop vs EarlyMerge
    c1 = compute_contrast(bias_df, 'HyperloopBERT', 'EarlyMergeHyperloopBERT', metric, primary_bands)
    
    contrasts = []
    if 'base' in c1:
        contrasts.append(c1['base'])
        
    # Decision
    decision = decision_logic(contrasts, disagreement_df)
    logger.info(f"=== GO/NO-GO VERDICT: {decision} ===")
    
    # Save stats
    stats_dir = os.path.join(results_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    if contrasts:
        pd.DataFrame(contrasts).to_csv(os.path.join(stats_dir, 'exploratory_stats.csv'), index=False)
        
    # Outline
    write_paper_outline(results_dir, contrasts, decision, disagreement_df)
    
    # Figures
    if ablation_df is not None and not ablation_df.empty:
        band = primary_bands.get('base')
        if band:
            plot_stream_ablation(ablation_df, metric, band, os.path.join(figures_dir, 'stream_ablation.png'))
            
    if disagreement_df is not None and not disagreement_df.empty:
        plot_stream_disagreement(disagreement_df, os.path.join(figures_dir, 'stream_disagreement.png'))

    logger.info("Stage 3 Analysis complete.")

if __name__ == "__main__":
    main()

import os
import sys
import glob
import pandas as pd
import torch
from datetime import datetime
from transformers import PreTrainedTokenizerFast
import argparse

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.architectures import build_model, get_model_info
from common.bias_metrics import score_bias_pair, passes_quality_screen
from common.io_schemas import BIAS_EXAMPLE_COLUMNS, BIAS_SUMMARY_BASE_COLUMNS
from common.stats_engine import bootstrap_ci
import Stage1.config_stage1 as cfg

logger = setup_logging('eval_bias_stage1')

def extract_model_metadata(snapshot_dir):
    """
    Parse directory structure to extract model metadata.
    Expected: models/stage1/iso_band_models/VanillaBERT_tiny_seed42/band_4p00
           or models/stage1/token_marker_models/VanillaBERT_tiny_seed42/tokens_50000000
    """
    parts = snapshot_dir.split(os.sep)
    band_or_marker_dir = parts[-1]
    run_dir = parts[-2]
    type_dir = parts[-3]
    
    run_parts = run_dir.split('_')
    if len(run_parts) >= 3:
        arch = run_parts[0]
        size = run_parts[1]
        seed_str = run_parts[2]
        seed = int(seed_str.replace('seed', ''))
    else:
        return None
        
    band = None
    marker = None
    
    if type_dir == 'iso_band_models':
        band_str = band_or_marker_dir.replace('band_', '').replace('p', '.')
        try:
            band = float(band_str)
        except ValueError:
            pass
    elif type_dir == 'token_marker_models':
        marker_str = band_or_marker_dir.replace('tokens_', '')
        try:
            marker = int(marker_str)
        except ValueError:
            pass
            
    return {
        'Architecture': arch,
        'Model_Size': size,
        'Seed': seed,
        'Band': band,
        'Token_Marker': marker,
        'Run_ID': run_dir,
        'Snapshot_Type': type_dir
    }

def get_snapshot_mlm_quality(results_dir, arch, size, seed, band, marker):
    """Get the pseudo-perplexity from the MLM summary table to check quality screen."""
    summary_path = os.path.join(results_dir, 'mlm', 'summary_table.csv')
    if not os.path.exists(summary_path):
        return None
        
    df = pd.read_csv(summary_path)
    mask = (df['Architecture'] == arch) & (df['Model_Size'] == size) & (df['Seed'] == seed)
    
    if band is not None:
        mask = mask & (df['Band'] == band)
    elif marker is not None:
        mask = mask & (df['Token_Marker'] == marker)
        
    match = df[mask]
    if not match.empty:
        pp = match['Pseudo_Perplexity'].iloc[0]
        # Handle nan/inf
        try:
            return float(pp)
        except:
            return None
    return None

def process_dataset(df, dataset_name, model, tokenizer, device, meta, model_info, out_progress_path):
    """Score a dataset and save progress."""
    results = []
    
    # Check if we can resume
    start_idx = 0
    if os.path.exists(out_progress_path):
        try:
            existing_df = pd.read_csv(out_progress_path)
            if not existing_df.empty and 'Row_Index' in existing_df.columns:
                start_idx = existing_df['Row_Index'].max() + 1
                logger.info(f"Resuming {dataset_name} from index {start_idx}")
        except Exception:
            pass
            
    batch_results = []
    
    for idx, row in df.iterrows():
        if idx < start_idx:
            continue
            
        stereo = row['stereo']
        anti = row['anti']
        
        # Typically the dataset will have a 'bias_type' or 'category' column
        category = row.get('bias_type', row.get('category', 'unknown'))
        
        scores = score_bias_pair(model, tokenizer, stereo, anti, device, compute_ss=False)
        
        res_row = {
            'Row_Index': idx,
            'Dataset': dataset_name,
            'Category': category,
            'Sentence_Stereotypical': stereo,
            'Sentence_AntiStereotypical': anti,
            'PLL_Stereotypical': scores['PLL_Stereotypical'],
            'PLL_AntiStereotypical': scores['PLL_AntiStereotypical'],
            'SS_PLL_Stereotypical': None,
            'SS_PLL_AntiStereotypical': None,
            'Effect_Size': scores['Effect_Size'],
            'Stereotype_Preferred': scores['Stereotype_Preferred'],
            'Stage': cfg.STAGE,
            'Architecture': meta['Architecture'],
            'Model_Size': meta['Model_Size'],
            'Hidden_Size': model_info['Hidden_Size'],
            'Seed': meta['Seed'],
            'Unique_Parameters': model_info['Unique_Parameters'],
            'Total_Parameters': model_info['Total_Parameters'],
            'Effective_Depth': model_info['Effective_Depth'],
            'Shared_Ratio': model_info['Shared_Ratio'],
            'Validation_Loss': meta.get('Validation_Loss'), # Not strict requirement
            'Band': meta['Band'],
            'Token_Marker': meta['Token_Marker'],
            'External_Calibration': False,
            'Needs_Review': False,
            'Timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        batch_results.append(res_row)
        
        if len(batch_results) >= cfg.BIAS_EVAL_BATCH_SIZE:
            append_df = pd.DataFrame(batch_results)[BIAS_EXAMPLE_COLUMNS]
            mode = 'a' if os.path.exists(out_progress_path) else 'w'
            header = not os.path.exists(out_progress_path)
            append_df.to_csv(out_progress_path, mode=mode, header=header, index=False)
            results.extend(batch_results)
            batch_results = []
            
    if batch_results:
        append_df = pd.DataFrame(batch_results)[BIAS_EXAMPLE_COLUMNS]
        mode = 'a' if os.path.exists(out_progress_path) else 'w'
        header = not os.path.exists(out_progress_path)
        append_df.to_csv(out_progress_path, mode=mode, header=header, index=False)
        results.extend(batch_results)
        
    # Read full file back for summary
    if os.path.exists(out_progress_path):
        return pd.read_csv(out_progress_path)
    return pd.DataFrame()

def compute_summary(df, meta, model_info, summary_path):
    """Compute summary stats and append to summary CSV."""
    if df.empty:
        return
        
    dataset = df['Dataset'].iloc[0]
    
    # Quality screen logic should have been applied before calling this, 
    # but we assume df has valid scores here.
    valid_df = df.dropna(subset=['Effect_Size', 'Stereotype_Preferred'])
    if valid_df.empty:
        return
        
    # Overall
    overall_pref = valid_df['Stereotype_Preferred'].mean()
    mean_effect = valid_df['Effect_Size'].mean()
    
    # Bootstrap CI
    _, ci_low, ci_high = bootstrap_ci(valid_df['Effect_Size'].tolist(), seed=meta['Seed'])
    
    # Macro average (average over categories)
    cat_means = valid_df.groupby('Category')['Stereotype_Preferred'].mean()
    macro_pref = cat_means.mean() if not cat_means.empty else overall_pref
    
    summary_row = {
        'Stage': cfg.STAGE,
        'Architecture': meta['Architecture'],
        'Model_Size': meta['Model_Size'],
        'Hidden_Size': model_info['Hidden_Size'],
        'Seed': meta['Seed'],
        'Unique_Parameters': model_info['Unique_Parameters'],
        'Total_Parameters': model_info['Total_Parameters'],
        'Effective_Depth': model_info['Effective_Depth'],
        'Shared_Ratio': model_info['Shared_Ratio'],
        'Band': meta['Band'],
        'Token_Marker': meta['Token_Marker'],
        'Overall_Stereotype_Preference_Rate': overall_pref,
        'Macro_Average_Preference_Rate': macro_pref,
        'Mean_Effect_Size': mean_effect,
        'Bootstrap_CI_Low': ci_low,
        'Bootstrap_CI_High': ci_high,
        'PLL_SS_PLL_Agreement': None, # Stage 2 feature
        'Timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    # Expand with dataset-specific columns
    df_sum = pd.DataFrame([summary_row])
    # For now, just save base columns. Stage 1 doesn't add many specific summary columns
    # beyond the base ones per dataset.
    
    for col in BIAS_SUMMARY_BASE_COLUMNS:
        if col not in df_sum.columns:
            df_sum[col] = None
    df_sum = df_sum[BIAS_SUMMARY_BASE_COLUMNS]
    
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    mode = 'a' if os.path.exists(summary_path) else 'w'
    header = not os.path.exists(summary_path)
    df_sum.to_csv(summary_path, mode=mode, header=header, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
    parser.add_argument('--resume', action='store_true', help='Resume existing eval')
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage1.py directly instead.")
        return
        
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, cfg.RESULTS_DIR)
    models_dir = os.path.join(base_dir, cfg.MODELS_DIR)
    
    # Load tokenzier
    tokenizer_dir = os.path.join(data_dir, 'tokenizer')
    if not os.path.exists(tokenizer_dir):
        logger.error(f"Tokenizer not found at {tokenizer_dir}")
        return
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    # Load datasets
    eval_dir = os.path.join(data_dir, 'datasets_eval')
    multicrows_path = os.path.join(eval_dir, 'multicrows', 'crows_pair_english.csv')
    indian_path = os.path.join(eval_dir, 'indian_bias', 'indian_bias_english.csv')
    
    datasets_to_eval = {}
    if os.path.exists(multicrows_path):
        datasets_to_eval['multicrows'] = pd.read_csv(multicrows_path)
    if os.path.exists(indian_path):
        datasets_to_eval['indian_bias'] = pd.read_csv(indian_path)
        
    if not datasets_to_eval:
        logger.error("No evaluation datasets found!")
        return
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Find all checkpoints
    checkpoints = []
    for snapshot_type in ['iso_band_models', 'token_marker_models']:
        type_dir = os.path.join(models_dir, snapshot_type)
        if not os.path.exists(type_dir):
            continue
        for root, dirs, files in os.walk(type_dir):
            if 'pytorch_model.bin' in files:
                checkpoints.append(root)
                
    for cp_dir in checkpoints:
        meta = extract_model_metadata(cp_dir)
        if not meta:
            continue
            
        # Quality screen
        pp = get_snapshot_mlm_quality(results_dir, meta['Architecture'], meta['Model_Size'], 
                                      meta['Seed'], meta['Band'], meta['Token_Marker'])
                                      
        if not passes_quality_screen(pp, cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD):
            logger.info(f"Skipping {cp_dir} - failed quality screen (PP={pp})")
            continue
            
        meta['Validation_Loss'] = None # We could fetch this from the summary, but it's not critical for eval execution
        
        logger.info(f"Evaluating model: {cp_dir}")
        
        # Build and load model
        model = build_model(meta['Architecture'], meta['Model_Size'])
        model_info = get_model_info(model)
        
        try:
            model.load_state_dict(torch.load(os.path.join(cp_dir, 'pytorch_model.bin'), map_location='cpu', weights_only=True))
        except Exception as e:
            logger.error(f"Failed to load model from {cp_dir}: {e}")
            continue
            
        model.to(device)
        model.eval()
        
        for ds_name, df in datasets_to_eval.items():
            # Build paths
            band_or_marker = f"band{meta['Band']}" if meta['Band'] else f"tokens{meta['Token_Marker']}"
            prog_filename = f"{ds_name}_{meta['Architecture']}_{meta['Model_Size']}_seed{meta['Seed']}_{band_or_marker}_progress.csv"
            out_progress_path = os.path.join(results_dir, 'bias', prog_filename)
            summary_path = os.path.join(results_dir, 'bias', f"{ds_name}_summary.csv")
            
            # Check if summary already exists for this snapshot
            if args.resume and os.path.exists(summary_path):
                sum_df = pd.read_csv(summary_path)
                
                mask = (sum_df['Architecture'] == meta['Architecture']) & \
                       (sum_df['Model_Size'] == meta['Model_Size']) & \
                       (sum_df['Seed'] == meta['Seed'])
                if meta['Band'] is not None:
                    mask = mask & (sum_df['Band'] == meta['Band'])
                else:
                    mask = mask & (sum_df['Token_Marker'] == meta['Token_Marker'])
                    
                if not sum_df[mask].empty:
                    logger.info(f"Skipping {ds_name} for {prog_filename} - summary already exists.")
                    continue
            
            logger.info(f"Running {ds_name} on {prog_filename}...")
            res_df = process_dataset(df, ds_name, model, tokenizer, device, meta, model_info, out_progress_path)
            
            logger.info(f"Computing summary for {ds_name}...")
            compute_summary(res_df, meta, model_info, summary_path)
            
        # Free memory
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

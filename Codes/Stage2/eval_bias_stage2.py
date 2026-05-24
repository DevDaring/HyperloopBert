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
from common.bias_metrics import score_bias_pair, score_winobias, passes_quality_screen
from common.io_schemas import BIAS_EXAMPLE_COLUMNS, BIAS_SUMMARY_BASE_COLUMNS, WINOBIAS_SUMMARY_COLUMNS
from common.stats_engine import bootstrap_ci
import Stage2.config_stage2 as cfg
from Stage1.eval_bias_stage1 import extract_model_metadata, get_snapshot_mlm_quality

logger = setup_logging('eval_bias_stage2')

def process_dataset(df, dataset_name, model, tokenizer, device, meta, model_info, out_progress_path):
    """Score a dataset and save progress (includes SS-PLL)."""
    results = []
    
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
        category = row.get('bias_type', row.get('category', 'unknown'))
        
        # Enable SS-PLL for Stage 2
        scores = score_bias_pair(model, tokenizer, stereo, anti, device, compute_ss=True)
        
        res_row = {
            'Row_Index': idx,
            'Dataset': dataset_name,
            'Category': category,
            'Sentence_Stereotypical': stereo,
            'Sentence_AntiStereotypical': anti,
            'PLL_Stereotypical': scores['PLL_Stereotypical'],
            'PLL_AntiStereotypical': scores['PLL_AntiStereotypical'],
            'SS_PLL_Stereotypical': scores['SS_PLL_Stereotypical'],
            'SS_PLL_AntiStereotypical': scores['SS_PLL_AntiStereotypical'],
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
            'Validation_Loss': meta.get('Validation_Loss'), 
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
        
    if os.path.exists(out_progress_path):
        return pd.read_csv(out_progress_path)
    return pd.DataFrame()

def compute_summary(df, meta, model_info, summary_path):
    """Compute summary stats and append to summary CSV."""
    if df.empty:
        return
        
    dataset = df['Dataset'].iloc[0]
    valid_df = df.dropna(subset=['Effect_Size', 'Stereotype_Preferred'])
    if valid_df.empty:
        return
        
    overall_pref = valid_df['Stereotype_Preferred'].mean()
    mean_effect = valid_df['Effect_Size'].mean()
    _, ci_low, ci_high = bootstrap_ci(valid_df['Effect_Size'].tolist(), seed=meta['Seed'])
    
    cat_means = valid_df.groupby('Category')['Stereotype_Preferred'].mean()
    macro_pref = cat_means.mean() if not cat_means.empty else overall_pref
    
    # Compute agreement between PLL and SS-PLL preferred directions
    ss_agreement = None
    if 'SS_PLL_Stereotypical' in valid_df.columns and not valid_df['SS_PLL_Stereotypical'].isnull().all():
        ss_valid = valid_df.dropna(subset=['SS_PLL_Stereotypical', 'SS_PLL_AntiStereotypical'])
        if not ss_valid.empty:
            pll_pref = ss_valid['PLL_Stereotypical'] > ss_valid['PLL_AntiStereotypical']
            ss_pref = ss_valid['SS_PLL_Stereotypical'] > ss_valid['SS_PLL_AntiStereotypical']
            ss_agreement = (pll_pref == ss_pref).mean()
    
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
        'PLL_SS_PLL_Agreement': ss_agreement,
        'Timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    df_sum = pd.DataFrame([summary_row])[BIAS_SUMMARY_BASE_COLUMNS]
    
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    mode = 'a' if os.path.exists(summary_path) else 'w'
    header = not os.path.exists(summary_path)
    df_sum.to_csv(summary_path, mode=mode, header=header, index=False)

def eval_winobias(model, tokenizer, device, winobias_dir, meta, model_info, summary_path):
    """Evaluate WinoBias and save summary."""
    pro_path1 = os.path.join(winobias_dir, 'winobias_type1_pro.csv')
    anti_path1 = os.path.join(winobias_dir, 'winobias_type1_anti.csv')
    
    if not (os.path.exists(pro_path1) and os.path.exists(anti_path1)):
        logger.warning(f"WinoBias data missing in {winobias_dir}")
        return
        
    pro_df = pd.read_csv(pro_path1)
    anti_df = pd.read_csv(anti_path1)
    
    # We expect WinoBias CSVs to have columns that allow us to formulate two options.
    # WinoBias HuggingFace dataset gives: document, tokens, coreference_clusters
    # For a simple PLL evaluation, we might need a specific prompt template, 
    # but score_winobias expects [(opt1, opt2), ...].
    # Assuming standard preprocessing to yield (correct_sentence, incorrect_sentence)
    # If the CSV doesn't have these, we'd need to generate them. For this pipeline,
    # we assume the CSV has 'correct_sentence' and 'incorrect_sentence'.
    
    pro_sentences = []
    if 'correct_sentence' in pro_df.columns and 'incorrect_sentence' in pro_df.columns:
        for _, row in pro_df.iterrows():
            pro_sentences.append(((row['correct_sentence'], row['incorrect_sentence']), 0))
            
    anti_sentences = []
    if 'correct_sentence' in anti_df.columns and 'incorrect_sentence' in anti_df.columns:
        for _, row in anti_df.iterrows():
            anti_sentences.append(((row['correct_sentence'], row['incorrect_sentence']), 0))
            
    if not pro_sentences or not anti_sentences:
        logger.warning("WinoBias CSV format not recognized for simple PLL eval. Skipping.")
        return
        
    logger.info("Scoring WinoBias...")
    scores = score_winobias(model, tokenizer, pro_sentences, anti_sentences, device)
    
    res_row = {
        'Stage': cfg.STAGE,
        'Architecture': meta['Architecture'],
        'Model_Size': meta['Model_Size'],
        'Hidden_Size': model_info['Hidden_Size'],
        'Seed': meta['Seed'],
        'Unique_Parameters': model_info['Unique_Parameters'],
        'Total_Parameters': model_info['Total_Parameters'],
        'Effective_Depth': model_info['Effective_Depth'],
        'Shared_Ratio': model_info['Shared_Ratio'],
        'Pro_Stereotype_Accuracy': scores['Pro_Stereotype_Accuracy'],
        'Anti_Stereotype_Accuracy': scores['Anti_Stereotype_Accuracy'],
        'Pro_Anti_Gap': scores['Pro_Anti_Gap'],
        'Timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    df_sum = pd.DataFrame([res_row])[WINOBIAS_SUMMARY_COLUMNS]
    
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
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage2.py directly instead.")
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
    indian_path = os.path.join(eval_dir, 'indian_bias', 'indian_bias_english.csv')
    winobias_dir = os.path.join(eval_dir, 'winobias')
    
    datasets_to_eval = {}
    if os.path.exists(multicrows_path):
        datasets_to_eval['multicrows'] = pd.read_csv(multicrows_path)
    if os.path.exists(indian_path):
        datasets_to_eval['indian_bias'] = pd.read_csv(indian_path)
        
    if not datasets_to_eval:
        logger.error("No evaluation datasets found!")
        return
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
            
        pp = get_snapshot_mlm_quality(results_dir, meta['Architecture'], meta['Model_Size'], 
                                      meta['Seed'], meta['Band'], meta['Token_Marker'])
                                      
        if not passes_quality_screen(pp, cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD):
            logger.info(f"Skipping {cp_dir} - failed quality screen (PP={pp})")
            continue
            
        meta['Validation_Loss'] = None 
        
        logger.info(f"Evaluating model: {cp_dir}")
        
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
            band_or_marker = f"band{meta['Band']}" if meta['Band'] else f"tokens{meta['Token_Marker']}"
            prog_filename = f"{ds_name}_{meta['Architecture']}_{meta['Model_Size']}_seed{meta['Seed']}_{band_or_marker}_progress.csv"
            out_progress_path = os.path.join(results_dir, 'bias', prog_filename)
            summary_path = os.path.join(results_dir, 'bias', f"{ds_name}_summary.csv")
            
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
            
        # WinoBias
        wino_sum_path = os.path.join(results_dir, 'bias', 'winobias_summary.csv')
        # Skip logic
        skip_wino = False
        if args.resume and os.path.exists(wino_sum_path):
            sum_df = pd.read_csv(wino_sum_path)
            mask = (sum_df['Architecture'] == meta['Architecture']) & \
                   (sum_df['Model_Size'] == meta['Model_Size']) & \
                   (sum_df['Seed'] == meta['Seed'])
            if not sum_df[mask].empty:
                skip_wino = True
                
        if not skip_wino:
            eval_winobias(model, tokenizer, device, winobias_dir, meta, model_info, wino_sum_path)
            
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

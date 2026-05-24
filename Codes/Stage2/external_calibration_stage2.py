import os
import sys
import pandas as pd
import torch
from datetime import datetime
from transformers import AutoModelForMaskedLM, AutoTokenizer
import argparse

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.bias_metrics import score_bias_pair
from common.io_schemas import EXTERNAL_CALIBRATION_COLUMNS
from common.stats_engine import bootstrap_ci
import Stage2.config_stage2 as cfg

logger = setup_logging('external_calibration_stage2')

def eval_external_model(model_name, datasets, device, out_path):
    """Evaluate external HF model on bias datasets."""
    logger.info(f"Loading external model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return
        
    for ds_name, df in datasets.items():
        logger.info(f"Evaluating {model_name} on {ds_name}...")
        
        results = []
        for idx, row in df.iterrows():
            stereo = row['stereo']
            anti = row['anti']
            category = row.get('bias_type', row.get('category', 'unknown'))
            
            # Note: ModerBERT might need a specific pipeline, but AutoModelForMaskedLM handles standard MLM.
            # PLL requires the model to have MLM head.
            try:
                scores = score_bias_pair(model, tokenizer, stereo, anti, device, compute_ss=True)
                
                results.append({
                    'Category': category,
                    'Stereotype_Preferred': scores['Stereotype_Preferred'],
                    'Effect_Size': scores['Effect_Size']
                })
            except Exception as e:
                pass
                
        if not results:
            continue
            
        res_df = pd.DataFrame(results).dropna(subset=['Effect_Size', 'Stereotype_Preferred'])
        if res_df.empty:
            continue
            
        overall_pref = res_df['Stereotype_Preferred'].mean()
        mean_effect = res_df['Effect_Size'].mean()
        cat_means = res_df.groupby('Category')['Stereotype_Preferred'].mean()
        macro_pref = cat_means.mean()
        
        sum_row = {
            'Model_Name': model_name,
            'Dataset': ds_name,
            'Category': 'ALL',
            'Overall_Stereotype_Preference_Rate': overall_pref,
            'Macro_Average_Preference_Rate': macro_pref,
            'Mean_Effect_Size': mean_effect,
            'External_Calibration': True,
            'Timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        out_df = pd.DataFrame([sum_row])[EXTERNAL_CALIBRATION_COLUMNS]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mode = 'a' if os.path.exists(out_path) else 'w'
        header = not os.path.exists(out_path)
        out_df.to_csv(out_path, mode=mode, header=header, index=False)
        
        # Free memory
    del model
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage2.py directly instead.")
        return
        
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, cfg.RESULTS_DIR)
    
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
    out_path = os.path.join(results_dir, 'bias', 'external_calibration.csv')
    
    if os.path.exists(out_path):
        os.remove(out_path) # start fresh
        
    for model_name in cfg.EXTERNAL_MODELS:
        eval_external_model(model_name, datasets_to_eval, device, out_path)
        
    logger.info(f"External calibration complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()

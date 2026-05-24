import os
import sys
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast
import argparse

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
import Stage3.config_stage3 as cfg
from Stage2.eval_glue_stage2 import train_and_eval_glue
from Stage1.eval_bias_stage1 import extract_model_metadata, get_snapshot_mlm_quality
from common.architectures import build_model, get_model_info
from common.bias_metrics import passes_quality_screen
from common.io_schemas import GLUE_SUMMARY_COLUMNS
from datetime import datetime

logger = setup_logging('eval_glue_stage3')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
    parser.add_argument('--resume', action='store_true', help='Resume existing eval')
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
        return
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoints = []
    type_dir = os.path.join(models_dir, 'iso_band_models')
    if os.path.exists(type_dir):
        for root, dirs, files in os.walk(type_dir):
            if 'pytorch_model.bin' in files:
                checkpoints.append(root)
                
    summary_path = os.path.join(results_dir, 'glue', 'summary_table.csv')
    
    for cp_dir in checkpoints:
        meta = extract_model_metadata(cp_dir)
        if not meta:
            continue
            
        num_streams = cfg.DEFAULT_NUM_STREAMS
        if 'streams' in meta['Run_ID']:
            parts = meta['Run_ID'].split('_')
            for p in parts:
                if p.startswith('streams'):
                    try:
                        num_streams = int(p.replace('streams', ''))
                    except:
                        pass
                        
        pp = get_snapshot_mlm_quality(results_dir, meta['Architecture'], meta['Model_Size'], 
                                      meta['Seed'], meta['Band'], meta['Token_Marker'])
                                      
        if not passes_quality_screen(pp, cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD):
            continue
            
        temp_model = build_model(meta['Architecture'], meta['Model_Size'], num_streams=num_streams)
        model_info = get_model_info(temp_model)
        del temp_model
        
        task_metrics = {}
        for task in cfg.GLUE_TASKS:
            model_path = os.path.join(cp_dir, 'pytorch_model.bin')
            metrics = train_and_eval_glue(model_path, meta, model_info, task, tokenizer, device)
            task_metrics[task] = metrics
            
            glue_avg = sum(m['Accuracy'] for m in task_metrics.values()) / len(task_metrics)
            
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
                'Task': task,
                'Accuracy': metrics['Accuracy'],
                'F1': metrics['F1'],
                'GLUE_Average': glue_avg,
                'Timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            df_sum = pd.DataFrame([res_row])[GLUE_SUMMARY_COLUMNS]
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            mode = 'a' if os.path.exists(summary_path) else 'w'
            header = not os.path.exists(summary_path)
            df_sum.to_csv(summary_path, mode=mode, header=header, index=False)

if __name__ == "__main__":
    main()

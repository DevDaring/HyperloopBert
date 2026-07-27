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

# We can reuse Stage 2 bias eval logic directly
from Stage2.eval_bias_stage2 import main as stage2_bias_main

logger = setup_logging('eval_bias_stage3')

def main():
    # We patch cfg to point to Stage 3 config so stage2_bias_main uses Stage 3 dirs if we just call it?
    # Actually, python imports bind the module namespace. It's safer to just duplicate the main body or run it.
    # We'll just write the full logic to avoid cross-module state issues.
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
    parser.add_argument('--resume', action='store_true', help='Resume existing eval')
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage3.py directly instead.")
        return
        
    # Full body similar to eval_bias_stage2
    from Stage2.eval_bias_stage2 import process_dataset, compute_summary, eval_winobias
    from Stage1.eval_bias_stage1 import (
        extract_model_metadata,
        get_snapshot_mlm_quality,
        get_snapshot_validation_loss,
    )
    from common.architectures import build_model, get_model_info
    from common.bias_metrics import passes_quality_screen
    
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

        # Correct stage label (compute_summary defaults to the Stage 2 config
        # when reused across stages)
        meta['Stage'] = cfg.STAGE

        # Stream count / merge point parsed from the run directory name by
        # extract_model_metadata; default to the primary stream count for
        # Hyperloop-family runs trained without a suffix.
        is_hyperloop = 'Hyperloop' in meta['Architecture']
        if is_hyperloop and meta.get('Stream_Count') is None:
            meta['Stream_Count'] = cfg.DEFAULT_NUM_STREAMS

        pp = get_snapshot_mlm_quality(results_dir, meta['Architecture'], meta['Model_Size'],
                                      meta['Seed'], meta['Band'], meta['Token_Marker'],
                                      stream_count=meta.get('Stream_Count'),
                                      merge_at=meta.get('Merge_At'))

        if not passes_quality_screen(pp, cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD):
            logger.info(f"Skipping {cp_dir} - failed quality screen (PP={pp})")
            continue

        meta['Validation_Loss'] = get_snapshot_validation_loss(
            results_dir, meta['Architecture'], meta['Model_Size'],
            meta['Seed'], meta['Band'], meta['Token_Marker'],
            stream_count=meta.get('Stream_Count'), merge_at=meta.get('Merge_At'))

        logger.info(f"Evaluating model: {cp_dir}")
        build_kwargs = {}
        if is_hyperloop:
            build_kwargs['num_streams'] = meta['Stream_Count']
        if meta.get('Merge_At') is not None:
            build_kwargs['merge_at'] = meta['Merge_At']
        model = build_model(meta['Architecture'], meta['Model_Size'], **build_kwargs)
        model_info = get_model_info(model)

        try:
            model.load_state_dict(torch.load(os.path.join(cp_dir, 'pytorch_model.bin'),
                                             map_location='cpu', weights_only=True))
        except Exception as e:
            logger.error(f"Failed to load model from {cp_dir}: {e}")
            continue

        model.to(device)
        model.eval()

        for ds_name, df in datasets_to_eval.items():
            band_or_marker = f"band{meta['Band']}" if meta['Band'] else f"tokens{meta['Token_Marker']}"
            suffix = ''
            if meta.get('Stream_Count') is not None:
                suffix += f"_streams{meta['Stream_Count']}"
            if meta.get('Merge_At') is not None:
                suffix += f"_merge{meta['Merge_At']}"
            prog_filename = (f"{ds_name}_{meta['Architecture']}_{meta['Model_Size']}"
                             f"_seed{meta['Seed']}{suffix}_{band_or_marker}_progress.csv")
            out_progress_path = os.path.join(results_dir, 'bias', prog_filename)
            summary_path = os.path.join(results_dir, 'bias', f"{ds_name}_summary.csv")

            # Unconditional dedup keyed on the FULL snapshot identity,
            # including Stream_Count and Merge_At (ablation-aware).
            if os.path.exists(summary_path):
                sum_df = pd.read_csv(summary_path)
                mask = (sum_df['Architecture'] == meta['Architecture']) & \
                       (sum_df['Model_Size'] == meta['Model_Size']) & \
                       (sum_df['Seed'] == meta['Seed'])
                if meta['Band'] is not None:
                    mask = mask & (sum_df['Band'] == meta['Band'])
                else:
                    mask = mask & (sum_df['Token_Marker'] == meta['Token_Marker'])
                if 'Stream_Count' in sum_df.columns:
                    if meta.get('Stream_Count') is not None:
                        mask = mask & (sum_df['Stream_Count'] == meta['Stream_Count'])
                    else:
                        mask = mask & (sum_df['Stream_Count'].isna())
                if 'Merge_At' in sum_df.columns:
                    if meta.get('Merge_At') is not None:
                        mask = mask & (sum_df['Merge_At'] == meta['Merge_At'])
                    else:
                        mask = mask & (sum_df['Merge_At'].isna())
                if not sum_df[mask].empty:
                    logger.info(f"Skipping {ds_name} for {prog_filename} - summary already exists.")
                    continue

            res_df = process_dataset(df, ds_name, model, tokenizer, device, meta,
                                     model_info, out_progress_path)
            compute_summary(res_df, meta, model_info, summary_path)

        # WinoBias
        wino_sum_path = os.path.join(results_dir, 'bias', 'winobias_summary.csv')
        eval_winobias(model, tokenizer, device, winobias_dir, meta, model_info, wino_sum_path)

        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

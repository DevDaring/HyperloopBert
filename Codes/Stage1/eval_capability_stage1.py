"""
Stage1/eval_capability_stage1.py -- capability-gate evidence (legs 1 and 3).

The Stage 1 capability gate (analyze_stage1.capability_gate) needs three legs
before any bias contrast may be interpreted:

  Leg 1: GLUE screen pulled FORWARD from Stage 2 (spec amendment A.2.2):
         SST-2 + RTE fine-tuning accuracy, tested above chance with a
         one-sided exact binomial test. Produced here.
  Leg 2: baseline bias detectable on VanillaBERT base (bootstrap CI > 0.5).
         Produced by eval_bias_stage1.py (unchanged).
  Leg 3: gendered-coreference signal control: WinoBias masked-pronoun accuracy
         on the pro-stereotypical splits above chance. On pro items, stereotype
         knowledge and coreference ability point the SAME way, so this is the
         most sensitive detector that any gendered pronoun signal was learned
         at all. Produced here. Anti splits are also scored (reported only).

Both legs run on iso-band snapshots at BASE size only (the gate is defined at
base). Scoring forwards run in FP32 (common/bias_metrics.py).

Outputs:
  results/stage1/glue/summary_table.csv          (GLUE_SUMMARY_COLUMNS)
  results/stage1/bias/winobias_capability.csv    (WINOBIAS_CAPABILITY_COLUMNS)
"""

import os
import sys
import argparse
from datetime import datetime

import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.architectures import build_model, get_model_info
from common.bias_metrics import passes_quality_screen, score_winobias_masked_pronoun
from common.io_schemas import GLUE_SUMMARY_COLUMNS, WINOBIAS_CAPABILITY_COLUMNS
import Stage1.config_stage1 as cfg
from Stage1.eval_bias_stage1 import extract_model_metadata, get_snapshot_mlm_quality

logger = setup_logging('eval_capability_stage1')

WINOBIAS_SPLITS = ['type1_pro', 'type2_pro', 'type1_anti', 'type2_anti']


def _append_rows(rows, columns, out_path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = not os.path.exists(out_path)
    df.to_csv(out_path, mode='a' if not header else 'w', header=header, index=False)


def _base_iso_checkpoints(models_dir):
    """Iso-band snapshots at base size (both architectures, all seeds)."""
    checkpoints = []
    type_dir = os.path.join(models_dir, 'iso_band_models')
    if not os.path.exists(type_dir):
        return checkpoints
    for root, dirs, files in os.walk(type_dir):
        if 'pytorch_model.bin' not in files:
            continue
        meta = extract_model_metadata(root)
        if not meta or meta['Band'] is None or meta['Model_Size'] != 'base':
            continue
        checkpoints.append((root, meta))
    return checkpoints


def run_glue_leg(checkpoints, results_dir, tokenizer, device, resume):
    """Leg 1: SST-2 + RTE on every base iso-band snapshot."""
    # Deferred import: reuses Stage 2's fine-tuning routine with Stage 1 config
    from Stage2.eval_glue_stage2 import train_and_eval_glue

    summary_path = os.path.join(results_dir, 'glue', 'summary_table.csv')

    for cp_dir, meta in checkpoints:
        pp = get_snapshot_mlm_quality(results_dir, meta['Architecture'],
                                      meta['Model_Size'], meta['Seed'],
                                      meta['Band'], meta['Token_Marker'])
        if not passes_quality_screen(pp, cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD):
            logger.info(f"Skipping GLUE for {cp_dir} - failed quality screen (PP={pp})")
            continue

        if resume and os.path.exists(summary_path):
            sum_df = pd.read_csv(summary_path)
            mask = (sum_df['Architecture'] == meta['Architecture']) & \
                   (sum_df['Model_Size'] == meta['Model_Size']) & \
                   (sum_df['Seed'] == meta['Seed']) & \
                   (sum_df['Band'] == meta['Band'])
            existing_tasks = sum_df[mask]['Task'].unique() if not sum_df[mask].empty else []
            if set(cfg.GLUE_TASKS).issubset(set(existing_tasks)):
                logger.info(f"Skipping GLUE for {cp_dir} - already complete.")
                continue

        temp_model = build_model(meta['Architecture'], meta['Model_Size'])
        model_info = get_model_info(temp_model)
        del temp_model

        task_metrics = {}
        for task in cfg.GLUE_TASKS:
            model_path = os.path.join(cp_dir, 'pytorch_model.bin')
            metrics = train_and_eval_glue(model_path, meta, model_info, task,
                                          tokenizer, device, glue_cfg=cfg)
            if metrics is None:
                continue  # failure already logged; never write a zero row
            task_metrics[task] = metrics
            glue_avg = sum(m['Accuracy'] for m in task_metrics.values()) / len(task_metrics)
            _append_rows([{
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
                'Stream_Count': meta.get('Stream_Count'),
                'Task': task,
                'Accuracy': metrics['Accuracy'],
                'F1': metrics['F1'],
                'Eval_Example_Count': metrics['N_Examples'],
                'GLUE_Average': glue_avg,
                'Timestamp': datetime.utcnow().isoformat() + 'Z',
            }], GLUE_SUMMARY_COLUMNS, summary_path)


def run_winobias_leg(checkpoints, results_dir, data_dir, tokenizer, device, resume):
    """Leg 3: WinoBias masked-pronoun counts on every base iso-band snapshot."""
    wb_dir = os.path.join(data_dir, 'datasets_eval', 'winobias')
    splits = {}
    for split in WINOBIAS_SPLITS:
        path = os.path.join(wb_dir, f'winobias_{split}.csv')
        if os.path.exists(path):
            splits[split] = pd.read_csv(path)
    if not splits:
        logger.error("No WinoBias CSVs found; run Dataset/download_eval_datasets.py "
                     "first. Capability leg 3 will be missing and the gate will "
                     "report INCOMPLETE.")
        return

    out_path = os.path.join(results_dir, 'bias', 'winobias_capability.csv')

    for cp_dir, meta in checkpoints:
        pp = get_snapshot_mlm_quality(results_dir, meta['Architecture'],
                                      meta['Model_Size'], meta['Seed'],
                                      meta['Band'], meta['Token_Marker'])
        if not passes_quality_screen(pp, cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD):
            logger.info(f"Skipping WinoBias for {cp_dir} - failed quality screen (PP={pp})")
            continue

        if resume and os.path.exists(out_path):
            done = pd.read_csv(out_path)
            mask = (done['Architecture'] == meta['Architecture']) & \
                   (done['Seed'] == meta['Seed']) & \
                   (done['Band'] == meta['Band'])
            if set(splits).issubset(set(done[mask]['Split'].unique())):
                logger.info(f"Skipping WinoBias for {cp_dir} - already complete.")
                continue

        model = build_model(meta['Architecture'], meta['Model_Size'])
        model_info = get_model_info(model)
        try:
            model.load_state_dict(torch.load(os.path.join(cp_dir, 'pytorch_model.bin'),
                                             map_location='cpu', weights_only=True))
        except Exception as e:
            logger.error(f"Failed to load model from {cp_dir}: {e}")
            continue
        model.to(device)
        model.eval()

        rows = []
        for split, df in splits.items():
            correct, scored = score_winobias_masked_pronoun(
                model, tokenizer, df, device, return_counts=True)
            if scored == 0:
                logger.warning(f"{meta['Architecture']} seed {meta['Seed']} "
                               f"band {meta['Band']}: {split} scored 0 items.")
                continue
            rows.append({
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
                'Split': split,
                'Correct_Count': correct,
                'Scored_Count': scored,
                'Accuracy': correct / scored,
                'Timestamp': datetime.utcnow().isoformat() + 'Z',
            })
            logger.info(f"{meta['Architecture']} seed {meta['Seed']} band "
                        f"{meta['Band']} {split}: {correct}/{scored} "
                        f"({correct / scored:.3f})")
        _append_rows(rows, WINOBIAS_CAPABILITY_COLUMNS, out_path)

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Skip completed snapshots')
    parser.add_argument('--skip-glue', action='store_true',
                        help='Only run the WinoBias leg')
    parser.add_argument('--skip-winobias', action='store_true',
                        help='Only run the GLUE leg')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, cfg.RESULTS_DIR)
    models_dir = os.path.join(base_dir, cfg.MODELS_DIR)

    tokenizer_dir = os.path.join(data_dir, 'tokenizer')
    if not os.path.exists(tokenizer_dir):
        logger.error(f"Tokenizer not found at {tokenizer_dir}")
        return
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoints = _base_iso_checkpoints(models_dir)
    if not checkpoints:
        logger.error("No base-size iso-band snapshots found. Train Stage 1 first.")
        return
    logger.info(f"Capability eval over {len(checkpoints)} base iso-band snapshots.")

    if not args.skip_winobias:
        run_winobias_leg(checkpoints, results_dir, data_dir, tokenizer, device, args.resume)
    if not args.skip_glue:
        run_glue_leg(checkpoints, results_dir, tokenizer, device, args.resume)

    logger.info("Capability evaluation complete. Run analyze_stage1.py for the gate verdict.")


if __name__ == "__main__":
    main()

"""
Stage1/qualitative_stage1.py -- paper-ready model-output dump for Stage 1.

Produces the human-readable "what does the MLM head actually predict" artifact
reviewers ask for, per trained snapshot that passes the quality screen:

  results/stage1/qualitative/mlm_topk_predictions.csv    (open-vocab top-k)
  results/stage1/qualitative/mlm_targeted_contrast.csv   (paired-token probs)
  results/stage1/qualitative/examples.md                 (paper-ready table)

Stage 2/3 reuse the same engine (common/qualitative_output.py); they only need
to point it at their own snapshots and results dir.
"""

import os
import sys
import argparse

import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.architectures import build_model, get_model_info
from common.bias_metrics import passes_quality_screen
from common.qualitative_output import dump_qualitative_output
import Stage1.config_stage1 as cfg
from Stage1.eval_bias_stage1 import extract_model_metadata, get_snapshot_mlm_quality

logger = setup_logging('qualitative_stage1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=10,
                        help='How many predicted tokens to record per probe')
    parser.add_argument('--sizes', nargs='+', default=cfg.SIZES,
                        help='Restrict to these sizes (default: all Stage 1 sizes)')
    parser.add_argument('--all-bands', action='store_true',
                        help='Dump every iso-band snapshot (default: quality-screened only)')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = os.path.join(results_dir, 'qualitative')
    # Fresh markdown each run so the paper table is not duplicated on re-runs.
    md_path = os.path.join(out_dir, 'examples.md')
    if os.path.exists(md_path):
        os.remove(md_path)
    os.makedirs(out_dir, exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Stage 1 -- Qualitative MLM-head output\n\n"
                "Each block shows one trained snapshot: the tokens the model "
                "predicts at a masked position (open vocabulary) and the "
                "probability it assigns to paired demographic tokens. All "
                "probabilities are FP32, the same precision as the primary PLL "
                "scorer. Read the log-odds column as the interpretable bias "
                "signal.\n")

    type_dir = os.path.join(models_dir, 'iso_band_models')
    if not os.path.exists(type_dir):
        logger.error(f"No iso-band snapshots under {type_dir}. Train Stage 1 first.")
        return

    n = 0
    for root, dirs, files in os.walk(type_dir):
        if 'pytorch_model.bin' not in files:
            continue
        meta = extract_model_metadata(root)
        if not meta or meta['Band'] is None or meta['Model_Size'] not in args.sizes:
            continue

        if not args.all_bands:
            pp = get_snapshot_mlm_quality(results_dir, meta['Architecture'],
                                          meta['Model_Size'], meta['Seed'],
                                          meta['Band'], meta['Token_Marker'])
            if not passes_quality_screen(pp, cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD):
                logger.info(f"Skipping {root} (failed quality screen, PP={pp}).")
                continue

        meta['Stage'] = cfg.STAGE
        model = build_model(meta['Architecture'], meta['Model_Size'])
        model_info = get_model_info(model)
        try:
            model.load_state_dict(torch.load(os.path.join(root, 'pytorch_model.bin'),
                                             map_location='cpu', weights_only=True))
        except Exception as e:
            logger.error(f"Failed to load {root}: {e}")
            continue
        model.to(device)
        model.eval()

        dump_qualitative_output(model, tokenizer, device, meta, model_info,
                                out_dir, topk=args.topk, logger=logger)
        n += 1
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    logger.info(f"Qualitative dump complete over {n} snapshots. See {md_path}")


if __name__ == "__main__":
    main()

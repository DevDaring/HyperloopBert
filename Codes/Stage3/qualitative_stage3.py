"""
Stage3/qualitative_stage3.py -- reviewer-facing MLM-output dump for Stage 3.

Produces the human-readable "what does the model actually predict" artifact for
EVERY Stage 3 architecture (VanillaBERT, LoopedBERT, ALBERTLoopedBERT,
HyperloopBERT), so the paper can show the model's behaviour side by side across
the weight-sharing spectrum rather than only reporting PLL numbers:

  results/stage3/qualitative/mlm_topk_predictions.csv    (open-vocab top-k)
  results/stage3/qualitative/mlm_targeted_contrast.csv   (paired-token probs)
  results/stage3/qualitative/examples.md                 (paper-ready table)

Uses the stage-agnostic engine in common/qualitative_output.py. All forwards run
in FP32, matching the primary PLL scorer.
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
from common.iso_loss import compute_primary_band
import Stage3.config_stage3 as cfg
from Stage1.eval_bias_stage1 import extract_model_metadata, get_snapshot_mlm_quality

logger = setup_logging('qualitative_stage3')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--all-bands', action='store_true',
                        help='Dump every snapshot (default: the common primary band only)')
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

    # Compare architectures at the SAME iso-loss band, so differences in the
    # printed predictions reflect architecture and not model quality.
    primary_band = None
    if not args.all_bands:
        import pandas as pd
        mlm_path = os.path.join(results_dir, 'mlm', 'summary_table.csv')
        mlm_df = pd.read_csv(mlm_path) if os.path.exists(mlm_path) else None
        primary_band = compute_primary_band(mlm_df, cfg.ARCHITECTURES, 'base',
                                            logger_obj=logger)
        logger.info(f"Common primary band for qualitative comparison: {primary_band}")
        if primary_band is None:
            logger.warning("No common primary band; falling back to all snapshots.")

    out_dir = os.path.join(results_dir, 'qualitative')
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, 'examples.md')
    if os.path.exists(md_path):
        os.remove(md_path)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Stage 3 -- Qualitative MLM-head output across the "
                "weight-sharing spectrum\n\n"
                "Each block is one trained architecture at the SAME iso-loss band "
                "(matched quality), showing the tokens the model predicts at a "
                "masked position and the probability it assigns to paired "
                "demographic tokens. Probabilities are FP32, matching the primary "
                "PLL scorer. The log-odds column is the interpretable bias "
                "signal.\n\n"
                "Construct scope: this is a CORRELATION-type measure "
                "(Wang et al. 2025, arXiv:2502.01926) -- stereotype ASSOCIATION, "
                "not difference-aware fairness.\n")

    type_dir = os.path.join(models_dir, 'iso_band_models')
    if not os.path.exists(type_dir):
        logger.error(f"No iso-band snapshots under {type_dir}.")
        return

    n = 0
    for root, dirs, files in os.walk(type_dir):
        if 'pytorch_model.bin' not in files:
            continue
        meta = extract_model_metadata(root)
        if not meta or meta['Band'] is None:
            continue
        if meta.get('Merge_At') is not None:
            continue
        if primary_band is not None and meta['Band'] != primary_band:
            continue

        pp = get_snapshot_mlm_quality(results_dir, meta['Architecture'],
                                      meta['Model_Size'], meta['Seed'],
                                      meta['Band'], meta['Token_Marker'],
                                      meta.get('Stream_Count'), meta.get('Merge_At'))
        if not passes_quality_screen(pp, cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD):
            logger.info(f"Skipping {root} (quality screen, PP={pp}).")
            continue

        meta['Stage'] = cfg.STAGE
        kwargs = {'num_streams': cfg.DEFAULT_NUM_STREAMS} \
            if meta['Architecture'] == 'HyperloopBERT' else {}
        model = build_model(meta['Architecture'], meta['Model_Size'], **kwargs)
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

    logger.info(f"Qualitative dump complete over {n} snapshots -> {md_path}")


if __name__ == "__main__":
    main()

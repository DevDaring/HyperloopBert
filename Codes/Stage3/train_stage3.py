import os
import sys
import json
import argparse
import types

import torch
from transformers import PreTrainedTokenizerFast

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.architectures import build_model, get_model_info
from common.train_loop import (
    seed_everything,
    prepare_validation_set,
    run_mlm_training,
    verify_training_data_integrity,
)
import Stage3.config_stage3 as cfg

logger = setup_logging('train_stage3')

# CITATION: Zeitoun, A., Torroba-Hennigen, L., & Kim, Y. (2026). Hyperloop
#           Transformers. arXiv:2604.21254.
#           [HyperloopBERT; ours = encoder-only adaptation + CWSA]
# CITATION: Zhu, D. et al. (2025). Hyper-Connections. ICLR 2025; Xie, Z. et al. (2025).
#           MHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880.
#           [hyper-connection basis + stability check, see stream_analysis_stage3.py]
# SUPPORT:  Zhu, R.-J. et al. (2025). Ouro. arXiv:2510.25741; Frey, M. et al. (2026).
#           arXiv:2603.08391. [SCH support; arXiv:2603.08391 is Frey 2026, not Zhu.]
# NOTE:     The stream-count ablation (n in {1,2,4}) is the causal dose-response;
#           n=1 is the collapse-to-Looped sanity check. EarlyMerge is an OOD
#           intervention, corroborating only, never labeled causal proof.


def build_run_list(seeds_to_run):
    """Primary runs (all architectures) + ablation runs (stream counts, early merge)."""
    runs = []
    for arch in cfg.ARCHITECTURES:
        for size in cfg.SIZES:
            for seed in seeds_to_run:
                runs.append({
                    'arch': arch, 'size': size, 'seed': seed,
                    'num_streams': cfg.DEFAULT_NUM_STREAMS if 'Hyperloop' in arch else None,
                    'merge_at': None,
                    'max_tokens': cfg.MAX_TOKENS,
                    # Primary runs get the seq=256 adaptation tail (spec 2.2),
                    # unless disabled for budget reasons (config ENABLE_SEQ_TAIL).
                    'seq_tail': getattr(cfg, 'ENABLE_SEQ_TAIL', True),
                })

    # Stream-count ablation arms (n=1, n=2) train at the SAME token budget as
    # the primary n=4 run, so the dose-response varies ONLY stream count --
    # never budget. (Design decision: rather than retraining a second n=4 arm
    # at a reduced budget, all arms share the primary budget; cheaper and no
    # identity collision between two n=4 runs.)
    # EarlyMerge is NOT trained: it is applied at eval time to the trained
    # 4-stream model (Stage3/stream_analysis_stage3.py), per the spec's
    # "OOD intervention, no new training" framing.
    ablation_seeds = cfg.ABLATION_SEEDS[:len(seeds_to_run)]
    for ns in cfg.NUM_STREAMS_ABLATION:
        if ns == cfg.DEFAULT_NUM_STREAMS:
            continue  # n=4 is already the primary HyperloopBERT run
        for seed in ablation_seeds:
            runs.append({
                'arch': 'HyperloopBERT', 'size': 'base', 'seed': seed,
                'num_streams': ns, 'merge_at': None,
                'max_tokens': cfg.MAX_TOKENS,
                # Ablation arms are seq=128 only (spec 2.3): the dose-response
                # must vary ONLY stream count, so no adaptation tail here.
                'seq_tail': False,
            })
    return runs


def make_run_id(run):
    run_id = f"{run['arch']}_{run['size']}_seed{run['seed']}"
    if run['num_streams'] is not None:
        run_id += f"_streams{run['num_streams']}"
    if run['merge_at'] is not None:
        run_id += f"_merge{run['merge_at']}"
    return run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=len(cfg.DEFAULT_SEEDS), help='Number of seeds')
    parser.add_argument('--sizes', nargs='+', default=cfg.SIZES, help='Sizes to train')
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoints')
    parser.add_argument('--dataset-namespace', type=str, default=cfg.VAL_DATASET_NAMESPACE)
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage3.py directly instead.")
        return

    seeds_to_run = cfg.DEFAULT_SEEDS[:args.seeds]

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, cfg.RESULTS_DIR)
    models_dir = os.path.join(base_dir, cfg.MODELS_DIR)
    checkpoints_dir = os.path.join(base_dir, cfg.CHECKPOINTS_DIR)

    train_file = os.path.join(data_dir, 'fineweb-edu', 'train_filtered.jsonl')
    val_file = os.path.join(data_dir, 'fineweb-edu', 'validation.jsonl')
    tokenizer_dir = os.path.join(data_dir, 'tokenizer')

    if not os.path.exists(tokenizer_dir):
        logger.error(f"Tokenizer not found at {tokenizer_dir}")
        return
    if not os.path.exists(train_file):
        logger.error(f"Training corpus not found at {train_file}. Run the Dataset stage first.")
        return

    # Spec 1.6: integrity check before consuming data (hard-fail on mismatch)
    verify_training_data_integrity(data_dir, train_file, tokenizer_dir, val_file, logger)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    logger.info("Loading validation set (fixed masking, seq=128)...")
    val_data = prepare_validation_set(
        val_file, tokenizer, cfg.SEQ_LENGTH, cfg.VAL_SAMPLES,
        mlm_probability=cfg.MLM_PROBABILITY, mask_prob=cfg.MLM_MASK_PROB,
        random_prob=cfg.MLM_RANDOM_PROB, logger=logger,
    )
    if not val_data:
        logger.error("No validation data loaded. Aborting.")
        return

    # Phase-2 validation set at the adaptation seq length (spec 2.2). Prepared
    # lazily/once and reused; iso-loss on the seq=256 tail must be measured at
    # seq=256, not seq=128.
    val_data_tail = None
    if any(run.get('seq_tail') for run in build_run_list(seeds_to_run)):
        logger.info(f"Loading validation set (fixed masking, seq={cfg.SEQ_LENGTH_TAIL})...")
        val_data_tail = prepare_validation_set(
            val_file, tokenizer, cfg.SEQ_LENGTH_TAIL, cfg.VAL_SAMPLES,
            mlm_probability=cfg.MLM_PROBABILITY, mask_prob=cfg.MLM_MASK_PROB,
            random_prob=cfg.MLM_RANDOM_PROB, logger=logger,
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    pipeline_state_path = os.path.join(results_dir, 'pipeline_state.json')
    pipeline_state = {}
    if args.resume and os.path.exists(pipeline_state_path):
        with open(pipeline_state_path, 'r') as f:
            pipeline_state = json.load(f)

    summary_path = os.path.join(results_dir, 'mlm', 'summary_table.csv')

    for run in build_run_list(seeds_to_run):
        run_id = make_run_id(run)

        if pipeline_state.get(run_id, False) and args.resume:
            logger.info(f"Skipping {run_id} - already complete.")
            continue

        logger.info(f"--- Starting training for {run_id} ---")
        seed_everything(run['seed'])

        # Only Hyperloop-family constructors accept num_streams / merge_at
        kwargs = {}
        if run['num_streams'] is not None:
            kwargs['num_streams'] = run['num_streams']
        if run['merge_at'] is not None:
            kwargs['merge_at'] = run['merge_at']
        model = build_model(run['arch'], run['size'], **kwargs)
        model_info = get_model_info(model)
        logger.info(f"Model specs: {model_info}")
        model.to(device)

        # Phase 1: seq=128 (per-run token budget; ablations skip the tail)
        run_cfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
        run_cfg.MAX_TOKENS = run['max_tokens']

        run_mlm_training(
            model=model, run_id=run_id, arch=run['arch'], size=run['size'],
            seed=run['seed'], cfg=run_cfg, tokenizer=tokenizer,
            model_info=model_info, train_file=train_file, val_data=val_data,
            device=device, models_dir=models_dir, summary_path=summary_path,
            checkpoints_dir=checkpoints_dir, logger=logger, resume=args.resume,
            stream_count=run['num_streams'], merge_at=run['merge_at'],
            seq_length=cfg.SEQ_LENGTH,
        )

        pipeline_state[run_id] = True
        os.makedirs(os.path.dirname(pipeline_state_path), exist_ok=True)
        with open(pipeline_state_path, 'w') as f:
            json.dump(pipeline_state, f, indent=2)

        # Phase 2: seq=256 adaptation tail (spec 2.2), continuing from the
        # phase-1 weights held in `model`. Distinct run_id (_seq256) so the tail
        # snapshots, checkpoints, and pipeline_state entry never collide with
        # phase 1. A fresh short LR schedule is intentional for the adaptation.
        if run.get('seq_tail') and val_data_tail:
            tail_run_id = f"{run_id}_seq256"
            if pipeline_state.get(tail_run_id, False) and args.resume:
                logger.info(f"Skipping {tail_run_id} - already complete.")
            else:
                logger.info(f"--- Sequence-length adaptation tail for {tail_run_id} "
                            f"({cfg.TAIL_TOKENS/1e6:.0f}M @ seq={cfg.SEQ_LENGTH_TAIL}) ---")
                tail_cfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
                tail_cfg.MAX_TOKENS = cfg.TAIL_TOKENS
                tail_cfg.TOKEN_MARKERS = list(cfg.TAIL_TOKEN_MARKERS)
                tail_cfg.DEFAULT_ISO_BANDS = list(cfg.TAIL_ISO_BANDS)
                run_mlm_training(
                    model=model, run_id=tail_run_id, arch=run['arch'],
                    size=run['size'], seed=run['seed'], cfg=tail_cfg,
                    tokenizer=tokenizer, model_info=model_info,
                    train_file=train_file, val_data=val_data_tail,
                    device=device, models_dir=models_dir,
                    summary_path=summary_path, checkpoints_dir=checkpoints_dir,
                    logger=logger, resume=args.resume,
                    stream_count=run['num_streams'], merge_at=run['merge_at'],
                    seq_length=cfg.SEQ_LENGTH_TAIL,
                )
                pipeline_state[tail_run_id] = True
                with open(pipeline_state_path, 'w') as f:
                    json.dump(pipeline_state, f, indent=2)

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

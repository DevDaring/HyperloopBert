import os
import sys
import json
import argparse

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
import Stage2.config_stage2 as cfg

logger = setup_logging('train_stage2')

# CITATION: Lan, Z. et al. (2020). ALBERT. ICLR 2020. [ALBERT sharing; no factored embedding]
# CITATION: Devlin, J. et al. (2019). BERT. NAACL 2019.
# CITATION: Saunshi, N. et al. (2025). On the Power of Looped Transformers. arXiv.
# CITATION: Bae, S. et al. (2025). Mixture-of-Recursions. arXiv:2507.10524.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=len(cfg.DEFAULT_SEEDS), help='Number of seeds')
    parser.add_argument('--sizes', nargs='+', default=cfg.SIZES, help='Sizes to train')
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoints')
    parser.add_argument('--dataset-namespace', type=str, default=cfg.VAL_DATASET_NAMESPACE)
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage2.py directly instead.")
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

    logger.info("Loading validation set (fixed masking)...")
    val_data = prepare_validation_set(
        val_file, tokenizer, cfg.SEQ_LENGTH, cfg.VAL_SAMPLES,
        mlm_probability=cfg.MLM_PROBABILITY, mask_prob=cfg.MLM_MASK_PROB,
        random_prob=cfg.MLM_RANDOM_PROB, logger=logger,
    )
    if not val_data:
        logger.error("No validation data loaded. Aborting.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    pipeline_state_path = os.path.join(results_dir, 'pipeline_state.json')
    pipeline_state = {}
    if args.resume and os.path.exists(pipeline_state_path):
        with open(pipeline_state_path, 'r') as f:
            pipeline_state = json.load(f)

    summary_path = os.path.join(results_dir, 'mlm', 'summary_table.csv')

    # Confirmatory architectures + exploratory controls (e.g. the
    # parameter-matched VanillaBERT6, analysed outside the confirmatory family)
    all_archs = list(cfg.ARCHITECTURES) + list(getattr(cfg, 'EXPLORATORY_ARCHITECTURES', []))
    for arch in all_archs:
        for size in args.sizes:
            for seed in seeds_to_run:
                run_id = f"{arch}_{size}_seed{seed}"

                if pipeline_state.get(run_id, False) and args.resume:
                    logger.info(f"Skipping {run_id} - already complete.")
                    continue

                logger.info(f"--- Starting training for {run_id} ---")
                seed_everything(seed)

                model = build_model(arch, size)
                model_info = get_model_info(model)
                logger.info(f"Model specs: {model_info}")
                model.to(device)

                run_mlm_training(
                    model=model, run_id=run_id, arch=arch, size=size, seed=seed,
                    cfg=cfg, tokenizer=tokenizer, model_info=model_info,
                    train_file=train_file, val_data=val_data, device=device,
                    models_dir=models_dir, summary_path=summary_path,
                    checkpoints_dir=checkpoints_dir, logger=logger,
                    resume=args.resume,
                )

                pipeline_state[run_id] = True
                os.makedirs(os.path.dirname(pipeline_state_path), exist_ok=True)
                with open(pipeline_state_path, 'w') as f:
                    json.dump(pipeline_state, f, indent=2)

                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

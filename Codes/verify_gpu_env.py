"""
verify_gpu_env.py -- GPU-box verification probe (spec A.2.7).

Run this ONCE on the training box before spending GPU-hours. It checks the two
things that can only be verified with a real CUDA + FlashAttention install:

  1. FlashAttention-2 actually ENGAGES (the int32 cu_seqlens fix works): a
     padded forward runs and the recorded attention path is still 'flash'
     afterwards -- i.e. it did NOT silently demote to SDPA at runtime.
  2. The data-integrity check runs end-to-end against the manifest.

Exit code 0 = all requested checks passed; non-zero = at least one failed.

Usage:
    python verify_gpu_env.py                 # both checks
    python verify_gpu_env.py --skip-integrity
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common.logging_setup import setup_logging

logger = setup_logging('verify_gpu_env')


def check_flash_engages():
    """Build a small model, run a padded forward, confirm 'flash' survived."""
    import torch
    from common.architectures import build_model
    from common.attention import (
        get_attention_path, _FLASH_AVAILABLE, set_attention_path_for_new_build,
    )

    if not torch.cuda.is_available():
        logger.error("FLASH CHECK: no CUDA device. This probe must run on the "
                     "training box; SDPA/eager fallbacks are not what we verify.")
        return False
    if not _FLASH_AVAILABLE:
        logger.error("FLASH CHECK: flash_attn did not import. Install "
                     "flash-attn matching the torch/CUDA ABI, or accept the "
                     "SDPA path (and update the paper's environment section).")
        return False

    device = 'cuda'
    model = build_model('VanillaBERT', 'tiny').to(device)
    model.eval()
    set_attention_path_for_new_build()

    # Padded batch (mixed real/pad) exercises the varlen cu_seqlens path -- the
    # exact path the int32 fix protects.
    input_ids = torch.randint(0, 100, (4, 128), device=device)
    attention_mask = torch.ones(4, 128, dtype=torch.long, device=device)
    attention_mask[0, 64:] = 0      # sequence 0 half-padded
    attention_mask[1, 100:] = 0     # sequence 1 mostly real

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(input_ids=input_ids, attention_mask=attention_mask)

    path = get_attention_path()
    if path == 'flash':
        logger.info("FLASH CHECK PASSED: padded varlen forward ran and the "
                    "attention path is still 'flash' (no runtime demotion).")
        return True
    logger.error(f"FLASH CHECK FAILED: attention path is '{path}' after the "
                 f"forward -- FlashAttention demoted at runtime. Check the "
                 f"warning logged above for the failing call (often a "
                 f"cu_seqlens dtype/ABI mismatch).")
    return False


def check_integrity():
    """Run the data-integrity check end-to-end against the manifest."""
    from common.train_loop import verify_training_data_integrity

    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    train_file = os.path.join(data_dir, 'fineweb-edu', 'train_filtered.jsonl')
    val_file = os.path.join(data_dir, 'fineweb-edu', 'validation.jsonl')
    tokenizer_dir = os.path.join(data_dir, 'tokenizer')

    if not os.path.exists(os.path.join(data_dir, 'dataset_manifest.json')):
        logger.error("INTEGRITY CHECK: no dataset_manifest.json. Run "
                     "Dataset/validate_and_manifest.py first.")
        return False
    try:
        # Raises SystemExit on mismatch; returns None on success.
        verify_training_data_integrity(data_dir, train_file, tokenizer_dir,
                                       val_file, logger)
    except SystemExit as e:
        logger.error(f"INTEGRITY CHECK FAILED: {e}")
        return False
    logger.info("INTEGRITY CHECK PASSED.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-flash', action='store_true')
    parser.add_argument('--skip-integrity', action='store_true')
    args = parser.parse_args()

    ok = True
    if not args.skip_flash:
        ok = check_flash_engages() and ok
    if not args.skip_integrity:
        ok = check_integrity() and ok

    if ok:
        logger.info("=== GPU environment verification PASSED ===")
        sys.exit(0)
    logger.error("=== GPU environment verification FAILED (see errors above) ===")
    sys.exit(1)


if __name__ == "__main__":
    main()

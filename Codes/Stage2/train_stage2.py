import os
import sys
import json
import argparse
import random
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizerFast

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.architectures import build_model, get_model_info, apply_mlm_mask, get_attention_path
from common.iso_loss import IsoBandTracker
from common.io_schemas import MLM_SUMMARY_COLUMNS
import Stage2.config_stage2 as cfg
from Stage1.train_stage1 import seed_everything, get_lr_schedule, data_generator, prepare_validation_set, evaluate, save_mlm_summary

logger = setup_logging('train_stage2')

# CITATION: Lan, Z. et al. (2020). ALBERT. ICLR 2020. [ALBERT sharing; no factored embedding]
# CITATION: Devlin, J. et al. (2019). BERT. NAACL 2019.
# CITATION: Saunshi, N. et al. (2025). On the Power of Looped Transformers. arXiv. 
# CITATION: Bae, J. et al. (2025). Looped encoder adaptation.

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
    
    # Check data integrity (basic)
    train_file = os.path.join(data_dir, 'fineweb-edu', 'train_filtered.jsonl')
    val_file = os.path.join(data_dir, 'fineweb-edu', 'validation.jsonl')
    tokenizer_dir = os.path.join(data_dir, 'tokenizer')
    
    if not os.path.exists(tokenizer_dir):
        logger.error(f"Tokenizer not found at {tokenizer_dir}")
        return
        
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    logger.info("Loading validation set...")
    val_data = prepare_validation_set(val_file, tokenizer, cfg.SEQ_LENGTH, cfg.VAL_SAMPLES)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    pipeline_state_path = os.path.join(results_dir, 'pipeline_state.json')
    pipeline_state = {}
    if args.resume and os.path.exists(pipeline_state_path):
        with open(pipeline_state_path, 'r') as f:
            pipeline_state = json.load(f)
            
    summary_path = os.path.join(results_dir, 'mlm', 'summary_table.csv')
    
    for arch in cfg.ARCHITECTURES:
        for size in args.sizes:
            for seed in seeds_to_run:
                run_id = f"{arch}_{size}_seed{seed}"
                
                # Check if we can reuse Stage 1 checkpoints for Vanilla/Looped if token budget >= Stage 1
                # To be completely safe and avoid missing checkpoints, we train from scratch or resume from Stage 2.
                # Cross-stage reuse logic could be complex, we just stick to Stage 2 dir.
                
                if pipeline_state.get(run_id, False) and args.resume:
                    logger.info(f"Skipping {run_id} - already complete.")
                    continue
                    
                logger.info(f"--- Starting training for {run_id} ---")
                seed_everything(seed)
                
                model = build_model(arch, size)
                model_info = get_model_info(model)
                logger.info(f"Model specs: {model_info}")
                
                model.to(device)
                
                # Check attention path
                dummy_input = torch.zeros(1, cfg.SEQ_LENGTH, dtype=torch.long, device=device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    model(input_ids=dummy_input)
                attn_path = get_attention_path()
                logger.info(f"Active attention path: {attn_path}")
                
                # Setup optimizer & scheduler
                tokens_per_step = cfg.EFFECTIVE_BATCH_SIZE * cfg.SEQ_LENGTH
                total_steps = int(cfg.MAX_TOKENS / tokens_per_step)
                warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
                
                optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, 
                                  betas=cfg.ADAMW_BETAS, eps=cfg.ADAMW_EPS, 
                                  weight_decay=cfg.WEIGHT_DECAY)
                scheduler = get_lr_schedule(optimizer, warmup_steps, total_steps)
                
                # Setup tracker
                iso_tracker = IsoBandTracker(
                    target_bands=cfg.DEFAULT_ISO_BANDS,
                    save_dir=os.path.join(models_dir, 'iso_band_models', run_id),
                    logger=logger
                )
                
                token_markers = cfg.TOKEN_MARKERS.copy()
                
                step = 0
                tokens_processed = 0
                micro_batch_size = cfg.MICRO_BATCH_SIZE
                accum_steps = max(1, cfg.EFFECTIVE_BATCH_SIZE // micro_batch_size)
                
                # Training loop
                model.train()
                gen = data_generator(train_file, tokenizer, cfg.SEQ_LENGTH, micro_batch_size)
                
                start_time = time.time()
                
                while tokens_processed < cfg.MAX_TOKENS:
                    try:
                        batch = next(gen)
                    except StopIteration:
                        logger.warning("Ran out of training data before token budget!")
                        break
                        
                    input_ids = batch.to(device)
                    
                    masked_input_ids, labels = apply_mlm_mask(
                        input_ids, tokenizer, 
                        prob=cfg.MLM_PROBABILITY, mask_prob=cfg.MLM_MASK_PROB,
                        random_prob=cfg.MLM_RANDOM_PROB, keep_prob=cfg.MLM_KEEP_PROB
                    )
                    
                    attention_mask = (input_ids != tokenizer.pad_token_id).long()
                    
                    try:
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
                            
                        logits = outputs.get('mlm_logits', outputs.get('logits'))
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        loss = loss_fct(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
                        
                        loss = loss / accum_steps
                        loss.backward()
                        
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if micro_batch_size > 4:
                            micro_batch_size //= 2
                            accum_steps = max(1, cfg.EFFECTIVE_BATCH_SIZE // micro_batch_size)
                            logger.warning(f"OOM. Halved micro-batch size to {micro_batch_size}.")
                            gen = data_generator(train_file, tokenizer, cfg.SEQ_LENGTH, micro_batch_size)
                            continue
                        else:
                            logger.error("OOM even at minimum micro-batch size!")
                            raise
                            
                    tokens_processed += input_ids.numel()
                    
                    if (step + 1) % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        actual_step = (step + 1) // accum_steps
                        
                        # Validate and check iso-bands
                        if actual_step % cfg.VAL_EVERY_STEPS == 0:
                            val_loss, p_perp, mask_acc = evaluate(model, val_data, tokenizer, device)
                            elapsed = time.time() - start_time
                            tps = tokens_processed / elapsed
                            
                            logger.info(f"Step {actual_step} | Tokens {tokens_processed/1e6:.1f}M | "
                                        f"Val Loss: {val_loss:.4f} | PP: {p_perp:.2f} | TPS: {tps:.1f}")
                                        
                            def save_snapshot(save_path):
                                model_path = os.path.join(save_path, "pytorch_model.bin")
                                torch.save(model.state_dict(), model_path)
                                
                            crossed_bands = iso_tracker.update(actual_step, val_loss, save_snapshot)
                            
                            for band in crossed_bands:
                                result = {
                                    'Stage': cfg.STAGE, 'Architecture': arch, 'Model_Size': size,
                                    'Hidden_Size': model_info['Hidden_Size'], 'Seed': seed,
                                    'Unique_Parameters': model_info['Unique_Parameters'],
                                    'Total_Parameters': model_info['Total_Parameters'],
                                    'Effective_Depth': model_info['Effective_Depth'],
                                    'Shared_Ratio': model_info['Shared_Ratio'],
                                    'Validation_Loss': val_loss, 'Pseudo_Perplexity': p_perp,
                                    'Mask_Accuracy': mask_acc, 'Tokens_Processed': tokens_processed,
                                    'Tokens_Per_Second': tps, 'GPU_Hours': elapsed / 3600.0,
                                    'Token_Marker': None, 'Band': band,
                                    'Timestamp': datetime.utcnow().isoformat() + 'Z'
                                }
                                save_mlm_summary(result, summary_path)
                                
                        # Check token markers
                        if token_markers and tokens_processed >= token_markers[0]:
                            marker = token_markers.pop(0)
                            marker_dir = os.path.join(models_dir, 'token_marker_models', run_id, f"tokens_{marker}")
                            os.makedirs(marker_dir, exist_ok=True)
                            torch.save(model.state_dict(), os.path.join(marker_dir, "pytorch_model.bin"))
                            
                            val_loss, p_perp, mask_acc = evaluate(model, val_data, tokenizer, device)
                            elapsed = time.time() - start_time
                            result = {
                                'Stage': cfg.STAGE, 'Architecture': arch, 'Model_Size': size,
                                'Hidden_Size': model_info['Hidden_Size'], 'Seed': seed,
                                'Unique_Parameters': model_info['Unique_Parameters'],
                                'Total_Parameters': model_info['Total_Parameters'],
                                'Effective_Depth': model_info['Effective_Depth'],
                                'Shared_Ratio': model_info['Shared_Ratio'],
                                'Validation_Loss': val_loss, 'Pseudo_Perplexity': p_perp,
                                'Mask_Accuracy': mask_acc, 'Tokens_Processed': tokens_processed,
                                'Tokens_Per_Second': tokens_processed / elapsed, 'GPU_Hours': elapsed / 3600.0,
                                'Token_Marker': marker, 'Band': None,
                                'Timestamp': datetime.utcnow().isoformat() + 'Z'
                            }
                            save_mlm_summary(result, summary_path)
                            
                    step += 1
                    
                pipeline_state[run_id] = True
                os.makedirs(os.path.dirname(pipeline_state_path), exist_ok=True)
                with open(pipeline_state_path, 'w') as f:
                    json.dump(pipeline_state, f, indent=2)
                    
                logger.info(f"Finished {run_id}")

if __name__ == "__main__":
    main()

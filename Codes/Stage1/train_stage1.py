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
from common.env_loader import env_loader
from common.architectures import build_model, get_model_info, apply_mlm_mask, get_attention_path
from common.iso_loss import IsoBandTracker
from common.io_schemas import MLM_SUMMARY_COLUMNS, make_empty_df
import Stage1.config_stage1 as cfg

logger = setup_logging('train_stage1')

# CITATION: Devlin, J. et al. (2019). BERT. NAACL 2019. [VanillaBERT]
# CITATION: Saunshi, N. et al. (2025). On the Power of Looped Transformers. arXiv. [SCH basis]
# CITATION: Bae, J. et al. (2025). Looped encoder adaptation. [LoopedBERT]
# COUNTER:  Zhu, L. et al. (2025). arXiv:2603.08391. [SCH counter-evidence; address, do not refute a priori]

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lr_schedule(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

def data_generator(filepath, tokenizer, seq_length, batch_size):
    """Generator that yields batches of tokenized sequences."""
    batch_input_ids = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get('text', '')
                if not text:
                    continue
                    
                tokens = tokenizer(text, max_length=seq_length, padding='max_length', 
                                   truncation=True, return_tensors='pt')
                
                batch_input_ids.append(tokens['input_ids'][0])
                
                if len(batch_input_ids) == batch_size:
                    yield torch.stack(batch_input_ids)
                    batch_input_ids = []
            except json.JSONDecodeError:
                continue

def prepare_validation_set(val_filepath, tokenizer, seq_length, max_samples):
    """Load validation set into memory."""
    input_ids = []
    
    if not os.path.exists(val_filepath):
        logger.warning(f"Validation file {val_filepath} not found.")
        return []
        
    with open(val_filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                obj = json.loads(line)
                text = obj.get('text', '')
                if text:
                    tokens = tokenizer(text, max_length=seq_length, padding='max_length', 
                                       truncation=True, return_tensors='pt')
                    input_ids.append(tokens['input_ids'][0])
            except json.JSONDecodeError:
                continue
                
    return input_ids

@torch.no_grad()
def evaluate(model, val_data, tokenizer, device, batch_size=64):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    
    for i in range(0, len(val_data), batch_size):
        batch = val_data[i:i+batch_size]
        if not batch:
            continue
            
        input_ids = torch.stack(batch).to(device)
        
        # Apply mask
        masked_input_ids, labels = apply_mlm_mask(
            input_ids, tokenizer, 
            prob=cfg.MLM_PROBABILITY, 
            mask_prob=cfg.MLM_MASK_PROB,
            random_prob=cfg.MLM_RANDOM_PROB,
            keep_prob=cfg.MLM_KEEP_PROB
        )
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            
        logits = outputs.get('mlm_logits', outputs.get('logits'))
        
        # Calculate loss manually to ignore index -100
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
        
        total_loss += loss.item() * input_ids.size(0)
        
        # Calculate mask accuracy
        mask = labels != -100
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == labels) & mask
        
        total_correct += correct.sum().item()
        total_masked += mask.sum().item()
        
    avg_loss = total_loss / len(val_data) if val_data else 0.0
    pseudo_perplexity = np.exp(avg_loss) if avg_loss < 50 else float('inf')
    mask_acc = total_correct / max(1, total_masked)
    
    model.train()
    return avg_loss, pseudo_perplexity, mask_acc

def save_mlm_summary(results, summary_path):
    """Append a result dict to the MLM summary CSV."""
    df = pd.DataFrame([results])
    
    # Ensure all required columns exist
    for col in MLM_SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = None
            
    df = df[MLM_SUMMARY_COLUMNS] # order columns
    
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    mode = 'a' if os.path.exists(summary_path) else 'w'
    header = not os.path.exists(summary_path)
    
    df.to_csv(summary_path, mode=mode, header=header, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=1, help='Number of seeds (1 or 3)')
    parser.add_argument('--sizes', nargs='+', default=cfg.SIZES, help='Sizes to train')
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoints')
    parser.add_argument('--dataset-namespace', type=str, default=cfg.VAL_DATASET_NAMESPACE)
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage1.py directly instead.")
        return
        
    seeds_to_run = cfg.DEFAULT_SEEDS[:args.seeds]
    if args.seeds == 3 and len(cfg.DEFAULT_SEEDS) >= 3:
        seeds_to_run = [42, 43, 44]
        
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
    if not val_data:
        logger.warning("No validation data loaded! Validation loss will be 0.")
        
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
                # Estimate total steps based on budget
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
                            # Need to recreate generator with new batch size
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
                                        
                            # Define callback for saving
                            def save_snapshot(save_path):
                                model_path = os.path.join(save_path, "pytorch_model.bin")
                                torch.save(model.state_dict(), model_path)
                                
                            crossed_bands = iso_tracker.update(actual_step, val_loss, save_snapshot)
                            
                            # Log to summary CSV
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
                            
                            # Get val loss for this marker
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

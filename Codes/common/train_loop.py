"""
common/train_loop.py

Shared MLM pre-training loop used by Stage 1, 2, and 3 training scripts.

Guarantees provided here (single source of truth so the stages cannot drift):
- FIXED validation masking: the validation set is masked ONCE with a dedicated
  RNG (seed independent of the training seed), so every architecture, size,
  and seed is validated against the identical masked token set. This matters
  because the iso-loss protocol keys snapshots off validation loss.
- Per-token validation loss: the reported validation loss is the average
  cross-entropy over masked tokens (not a per-sequence average of batch means).
- Document chunking: each training document is split into consecutive
  (seq_length - 2)-token chunks instead of truncating to a single chunk, so
  the token budget reflects text actually consumed.
- Non-padding token accounting: tokens_processed counts real tokens only.
- Mid-run checkpointing and resume: model, optimizer, scheduler, RNG states,
  iso-band tracker state, remaining token markers, and the training-data
  document cursor are saved every checkpoint_every_steps optimizer steps.
  On resume, training continues from the next document boundary after the
  last fully-consumed batch.
"""

# CITATION: Devlin, J. et al. (2019). BERT. NAACL 2019.  [MLM objective]
# PRE-REGISTERED ENDPOINT: The primary comparison is at matched validation loss
#           (iso-perplexity), NOT at fixed token budget.

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from common.architectures import apply_mlm_mask, get_attention_path
from common.iso_loss import IsoBandTracker
from common.io_schemas import MLM_SUMMARY_COLUMNS, ISO_CHECKPOINT_COLUMNS
from common.integrity import _calculate_md5, _count_rows

# Seed for the FIXED validation masking. Deliberately constant and independent
# of the per-run training seed: every (architecture, size, seed) is validated
# on the identical masked validation set.
VALIDATION_MASK_SEED: int = 1234

# Chunks shorter than this many body tokens are dropped (document tails).
_MIN_CHUNK_TOKENS: int = 16


def verify_training_data_integrity(data_dir: str, train_file: str,
                                   tokenizer_dir: str, val_file: str,
                                   logger) -> None:
    """
    Spec 1.6: verify data against the manifest BEFORE consuming it, and
    hard-fail on mismatch. Run once per training-script invocation.

    Targeted (not a full-corpus MD5, which would cost minutes per run):
      - tokenizer.json      -> MD5 must match (small; a silent tokenizer
                               rebuild would shift EVERY downstream number)
      - train / validation  -> existence + row count must match (fast; a
                               truncated or re-carved corpus is caught)

    If no manifest exists, warn and proceed (fresh setups before the manifest
    step have nothing to check against).
    """
    manifest_path = os.path.join(data_dir, 'dataset_manifest.json')
    if not os.path.exists(manifest_path):
        logger.warning("No dataset_manifest.json found; data-integrity check "
                       "skipped. Run Dataset/validate_and_manifest.py to enable it.")
        return

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    manifest_by_base = {os.path.basename(k): (k, v) for k, v in manifest.items()}
    failures = []

    def _entry(path):
        return manifest_by_base.get(os.path.basename(path))

    # Tokenizer: MD5 (small file, highest-risk silent shift)
    tok_json = os.path.join(tokenizer_dir, 'tokenizer.json')
    ent = _entry(tok_json)
    if ent and os.path.exists(tok_json):
        if _calculate_md5(tok_json) != ent[1]['md5_hash']:
            failures.append(f"tokenizer.json MD5 mismatch vs manifest ({ent[0]})")
    elif ent and not os.path.exists(tok_json):
        failures.append("tokenizer.json missing but present in manifest")

    # Corpus files: existence + row count (fast)
    for path, label in ((train_file, 'training corpus'), (val_file, 'validation set')):
        ent = _entry(path)
        if not ent:
            continue
        if not os.path.exists(path):
            failures.append(f"{label} missing: {path}")
            continue
        rows = _count_rows(path)
        if rows != ent[1]['row_count']:
            failures.append(f"{label} row count {rows} != manifest {ent[1]['row_count']} "
                            f"({path}) -- corpus changed since manifest was built")

    if failures:
        for msg in failures:
            logger.error(f"DATA INTEGRITY FAILURE: {msg}")
        raise SystemExit(
            "Aborting training: data integrity check failed. Re-run the Dataset "
            "stage (or Dataset/validate_and_manifest.py) so results stay comparable.")

    logger.info("Data integrity check PASSED (tokenizer MD5 + corpus row counts match manifest).")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_schedule(optimizer, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return LambdaLR(optimizer, lr_lambda)


def data_generator(filepath: str, tokenizer, seq_length: int, batch_size: int,
                   skip_docs: int = 0, doc_cursor: Optional[List[int]] = None):
    """
    Yield batches of (seq_length)-token input_id tensors.

    Documents are tokenized WITHOUT truncation and split into consecutive
    chunks of (seq_length - 2) body tokens, each wrapped in [CLS] ... [SEP]
    and padded to seq_length. Tails shorter than _MIN_CHUNK_TOKENS are dropped.

    skip_docs: number of leading documents to skip (resume fast-forward).
    doc_cursor: optional single-element list; doc_cursor[0] is kept at the
    index of the first document NOT yet fully consumed by a yielded batch.
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    max_body = seq_length - 2

    batch_input_ids: List[torch.Tensor] = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for doc_idx, line in enumerate(f):
            if doc_idx < skip_docs:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get('text', '')
            if not text:
                continue

            ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
            for start in range(0, len(ids), max_body):
                chunk = ids[start:start + max_body]
                if len(chunk) < _MIN_CHUNK_TOKENS:
                    break
                seq = [cls_id] + chunk + [sep_id]
                if len(seq) < seq_length:
                    seq = seq + [pad_id] * (seq_length - len(seq))
                batch_input_ids.append(torch.tensor(seq, dtype=torch.long))

                if len(batch_input_ids) == batch_size:
                    yield torch.stack(batch_input_ids)
                    batch_input_ids = []
                    if doc_cursor is not None:
                        # Conservative cursor: resume from this document. A
                        # resumed run restarts at the current document boundary,
                        # re-seeing at most the chunks of one document.
                        doc_cursor[0] = doc_idx


def memmap_generator(bin_path: str, seq_length: int, batch_size: int,
                     skip_blocks: int = 0, block_cursor: Optional[List[int]] = None,
                     loop: bool = True):
    """
    Yield batches from a pre-tokenized .bin produced by
    Dataset/pretokenize_corpus.py -- a flat uint16 array of fixed-size
    [CLS] ... [SEP] padded blocks with semantics identical to data_generator,
    but with ZERO tokenisation at train time.

    On-the-fly tokenisation measured ~4% model-FLOPs utilisation on an L4 (the
    single-threaded tokenizer starves the GPU); reading pre-tokenized blocks
    removes that bottleneck entirely.

    skip_blocks / block_cursor mirror the skip_docs / doc_cursor resume
    contract of data_generator. loop=True wraps around so a token budget larger
    than the corpus is served as multiple epochs (standard for BERT, which
    trained ~40 epochs).
    """
    arr = np.memmap(bin_path, dtype=np.uint16, mode='r')
    n_blocks = arr.shape[0] // seq_length
    arr = arr[:n_blocks * seq_length].reshape(n_blocks, seq_length)

    i = skip_blocks % n_blocks if n_blocks else 0
    while True:
        end = min(i + batch_size, n_blocks)
        batch = np.asarray(arr[i:end], dtype=np.int64)
        if batch.shape[0] > 0:
            yield torch.from_numpy(batch)
            if block_cursor is not None:
                block_cursor[0] = end
        i = end
        if i >= n_blocks:
            if not loop:
                return
            i = 0


def resolve_train_source(train_file: str, seq_length: int, logger=None):
    """
    Prefer the pre-tokenized binary when present (much faster), else fall back
    to on-the-fly tokenisation. Returns (kind, path) with kind in {'bin','jsonl'}.
    """
    candidate = os.path.join(os.path.dirname(train_file), f"train_{seq_length}.bin")
    if os.path.exists(candidate):
        if logger:
            logger.info(f"Using PRE-TOKENIZED corpus: {candidate}")
        return 'bin', candidate
    if logger:
        logger.warning(
            f"No pre-tokenized corpus at {candidate}; falling back to on-the-fly "
            f"tokenisation (measured ~4% MFU -- run Dataset/pretokenize_corpus.py "
            f"to get ~8x throughput).")
    return 'jsonl', train_file


def prepare_validation_set(val_filepath: str, tokenizer, seq_length: int,
                           max_samples: int, mlm_probability: float = 0.15,
                           mask_prob: float = 0.80, random_prob: float = 0.10,
                           batch_size: int = 64, logger=None) -> List[Dict[str, torch.Tensor]]:
    """
    Build the FIXED masked validation set.

    Returns a list of batch dicts with keys 'masked_input_ids', 'labels',
    'attention_mask' (CPU tensors). Masking uses a dedicated torch.Generator
    seeded with VALIDATION_MASK_SEED so it is identical for every model.
    """
    if not os.path.exists(val_filepath):
        if logger:
            logger.warning(f"Validation file {val_filepath} not found.")
        return []

    input_ids_list: List[torch.Tensor] = []
    with open(val_filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(input_ids_list) >= max_samples:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get('text', '')
            if not text:
                continue
            tokens = tokenizer(text, max_length=seq_length, padding='max_length',
                               truncation=True, return_tensors='pt')
            input_ids_list.append(tokens['input_ids'][0])

    if not input_ids_list:
        return []

    gen = torch.Generator()
    gen.manual_seed(VALIDATION_MASK_SEED)

    batches: List[Dict[str, torch.Tensor]] = []
    for i in range(0, len(input_ids_list), batch_size):
        input_ids = torch.stack(input_ids_list[i:i + batch_size])
        masked_input_ids, labels = apply_mlm_mask(
            input_ids, tokenizer,
            prob=mlm_probability, mask_prob=mask_prob, random_prob=random_prob,
            generator=gen,
        )
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        batches.append({
            'masked_input_ids': masked_input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        })
    if logger:
        n_masked = int(sum((b['labels'] != -100).sum().item() for b in batches))
        logger.info(f"Fixed masked validation set: {len(input_ids_list)} sequences, "
                    f"{n_masked} masked target tokens (mask seed {VALIDATION_MASK_SEED}).")
    return batches


@torch.no_grad()
def evaluate(model, val_data: List[Dict[str, torch.Tensor]], tokenizer, device: str):
    """
    Evaluate on the fixed masked validation set.

    Returns (avg_loss, pseudo_perplexity, mask_accuracy) where avg_loss is the
    mean cross-entropy over ALL masked tokens (per-token weighting).
    """
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_correct = 0
    total_masked = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    for batch in val_data:
        masked_input_ids = batch['masked_input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
        logits = outputs.get('mlm_logits', outputs.get('logits'))

        vocab = logits.size(-1)
        total_nll += loss_fct(logits.view(-1, vocab).float(), labels.view(-1)).item()

        mask = labels != -100
        preds = torch.argmax(logits, dim=-1)
        total_correct += ((preds == labels) & mask).sum().item()
        total_masked += mask.sum().item()

    if was_training:
        model.train()

    if total_masked == 0:
        return 0.0, float('inf'), 0.0

    avg_loss = total_nll / total_masked
    pseudo_perplexity = float(np.exp(avg_loss)) if avg_loss < 50 else float('inf')
    mask_acc = total_correct / total_masked
    return avg_loss, pseudo_perplexity, mask_acc


def save_mlm_summary(results: Dict, summary_path: str) -> None:
    """Append a result dict to the MLM summary CSV (canonical columns)."""
    df = pd.DataFrame([results])
    for col in MLM_SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[MLM_SUMMARY_COLUMNS]
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    header = not os.path.exists(summary_path)
    df.to_csv(summary_path, mode='a' if not header else 'w', header=header, index=False)


def _write_iso_index_row(summary_path: str, cfg, arch: str, size: str, seed: int,
                         model_info: Dict, stream_count, merge_at,
                         band: float, val_loss: float, crossed_step: int,
                         run_snapshot_dir: str) -> None:
    """
    Append a row to results/<stage>/iso_checkpoints/index.csv (spec-required):
    the durable evidence that each snapshot's ACTUAL validation loss matched
    its band, so reviewers can audit the iso-loss matching.
    """
    results_dir = os.path.dirname(os.path.dirname(summary_path))
    index_path = os.path.join(results_dir, 'iso_checkpoints', 'index.csv')
    band_str = f"{band:.2f}".replace('.', 'p')
    row = {
        'Stage': cfg.STAGE, 'Architecture': arch, 'Model_Size': size,
        'Hidden_Size': model_info['Hidden_Size'], 'Seed': seed,
        'Unique_Parameters': model_info['Unique_Parameters'],
        'Total_Parameters': model_info['Total_Parameters'],
        'Effective_Depth': model_info['Effective_Depth'],
        'Shared_Ratio': model_info['Shared_Ratio'],
        'Band': band,
        'Snapshot_Path': os.path.join(run_snapshot_dir, f'band_{band_str}'),
        'Validation_Loss_At_Snapshot': val_loss,
        'Crossed_At_Step': crossed_step,
        'Timestamp': datetime.utcnow().isoformat() + 'Z',
    }
    df = pd.DataFrame([row])
    for col in ISO_CHECKPOINT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[ISO_CHECKPOINT_COLUMNS]
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    header = not os.path.exists(index_path)
    df.to_csv(index_path, mode='a' if not header else 'w', header=header, index=False)


def _rng_state_dict() -> Dict:
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _load_rng_state(state: Dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'torch_cuda' in state:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def run_mlm_training(*, model, run_id: str, arch: str, size: str, seed: int,
                     cfg, tokenizer, model_info: Dict, train_file: str,
                     val_data: List[Dict[str, torch.Tensor]], device: str,
                     models_dir: str, summary_path: str, checkpoints_dir: str,
                     logger, resume: bool = False,
                     stream_count: Optional[int] = None,
                     merge_at: Optional[int] = None,
                     seq_length: Optional[int] = None) -> None:
    """
    Full single-run MLM training with iso-band snapshotting, token-marker
    snapshotting, and mid-run checkpoint/resume.

    seq_length: sequence length for this run; defaults to cfg.SEQ_LENGTH. The
    Stage 3 primary runs call this twice -- once at cfg.SEQ_LENGTH, then once at
    cfg.SEQ_LENGTH_TAIL for the sequence-length adaptation phase -- so the value
    is parameterised here rather than read straight from cfg. val_data must be
    masked at the SAME seq_length (the caller is responsible for that).
    """
    seq_length = seq_length if seq_length is not None else cfg.SEQ_LENGTH
    # Probe attention path once
    dummy_input = torch.zeros(1, seq_length, dtype=torch.long, device=device)
    with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
        model(input_ids=dummy_input)
    logger.info(f"Active attention path: {get_attention_path()}")

    # The token budget counts REAL (non-pad) tokens, but a step yields
    # EFFECTIVE_BATCH_SIZE * SEQ_LENGTH padded positions. Sizing the schedule
    # in padded tokens would drive LR to 0 before the real-token budget is
    # consumed. SEQ_FILL_RATIO is the expected real/padded fill after document
    # chunking (conservative: a slightly long schedule keeps LR > 0 to the end).
    fill_ratio = getattr(cfg, 'SEQ_FILL_RATIO', 0.90)
    tokens_per_step = cfg.EFFECTIVE_BATCH_SIZE * seq_length * fill_ratio
    total_steps = int(cfg.MAX_TOKENS / tokens_per_step)
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)

    # Gradient checkpointing (spec 2.4): 'auto' enables it for the base size
    # only -- tiny/small fit comfortably in 24 GB and would pay ~30% compute
    # for nothing. Set GRADIENT_CHECKPOINTING = True/False to force.
    gc_mode = getattr(cfg, 'GRADIENT_CHECKPOINTING', 'auto')
    if gc_mode is True or (gc_mode == 'auto' and size == 'base'):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing ENABLED for this run.")

    # Per-architecture peak LR. HyperloopBERT reaches the MLM breakthrough ~3x
    # earlier in tokens than the other arms, so the shared 3e-4 peak is far
    # hotter for it: at 3e-4 it diverged 3300 steps after warmup ended
    # (loss 2.5250 -> 6.86 -> NaN). Comparing an architecture at a setting that
    # breaks it is not a fair control, so peak LR is tuned per architecture and
    # the arms are matched on validation loss (iso-loss), not on hyperparameters.
    _lr = getattr(cfg, "LEARNING_RATE_OVERRIDES", {}).get(arch, cfg.LEARNING_RATE)
    if _lr != cfg.LEARNING_RATE:
        logger.info(f"Peak LR override for {arch}: {_lr} (default {cfg.LEARNING_RATE})")
    optimizer = AdamW(model.parameters(), lr=_lr,
                      betas=cfg.ADAMW_BETAS, eps=cfg.ADAMW_EPS,
                      weight_decay=cfg.WEIGHT_DECAY)
    scheduler = get_lr_schedule(optimizer, warmup_steps, total_steps)

    iso_tracker = IsoBandTracker(
        target_bands=cfg.DEFAULT_ISO_BANDS,
        save_dir=os.path.join(models_dir, 'iso_band_models', run_id),
        logger=logger,
    )

    token_markers = list(cfg.TOKEN_MARKERS)
    step = 0
    tokens_processed = 0
    nonfinite_grads = 0          # cumulative skipped updates
    consecutive_nonfinite = 0    # resets on any finite update
    skip_docs = 0
    micro_batch_size = cfg.MICRO_BATCH_SIZE

    checkpoint_every = getattr(cfg, 'CHECKPOINT_EVERY_STEPS', 2000)
    ckpt_dir = os.path.join(checkpoints_dir, run_id)
    ckpt_path = os.path.join(ckpt_dir, 'latest.pt')

    # ---- Resume mid-run if a checkpoint exists ----
    if resume and os.path.exists(ckpt_path):
        logger.info(f"Resuming {run_id} from mid-run checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        iso_tracker.load_state_dict(ckpt['iso_tracker'])
        token_markers = ckpt['token_markers']
        step = ckpt['step']
        tokens_processed = ckpt['tokens_processed']
        skip_docs = ckpt['docs_consumed']
        micro_batch_size = ckpt.get('micro_batch_size', micro_batch_size)
        _load_rng_state(ckpt['rng'])
        logger.info(f"Resumed at optimizer step {step // max(1, cfg.EFFECTIVE_BATCH_SIZE // micro_batch_size)}, "
                    f"{tokens_processed / 1e6:.1f}M tokens, doc cursor {skip_docs}.")

    accum_steps = max(1, cfg.EFFECTIVE_BATCH_SIZE // micro_batch_size)
    doc_cursor = [skip_docs]
    last_val_loss: Optional[float] = None

    model.train()
    # Prefer the pre-tokenized memmap corpus when available (identical sequence
    # semantics, ~8x throughput -- see Dataset/pretokenize_corpus.py).
    _src_kind, _src_path = resolve_train_source(train_file, seq_length, logger)

    def _make_gen(skip):
        if _src_kind == 'bin':
            return memmap_generator(_src_path, seq_length, micro_batch_size,
                                    skip_blocks=skip, block_cursor=doc_cursor)
        return data_generator(_src_path, tokenizer, seq_length, micro_batch_size,
                              skip_docs=skip, doc_cursor=doc_cursor)

    gen = _make_gen(skip_docs)
    start_time = time.time()
    start_tokens = tokens_processed

    def _identity_row(val_loss, p_perp, mask_acc, elapsed, band=None, marker=None):
        return {
            'Stage': cfg.STAGE, 'Architecture': arch, 'Model_Size': size,
            'Stream_Count': stream_count, 'Merge_At': merge_at,
            'Hidden_Size': model_info['Hidden_Size'], 'Seed': seed,
            'Unique_Parameters': model_info['Unique_Parameters'],
            'Total_Parameters': model_info['Total_Parameters'],
            'Effective_Depth': model_info['Effective_Depth'],
            'Shared_Ratio': model_info['Shared_Ratio'],
            'Validation_Loss': val_loss, 'Pseudo_Perplexity': p_perp,
            'Mask_Accuracy': mask_acc, 'Tokens_Processed': tokens_processed,
            'Tokens_Per_Second': (tokens_processed - start_tokens) / max(1e-9, elapsed),
            'GPU_Hours': elapsed / 3600.0,
            'Token_Marker': marker, 'Band': band,
            'Timestamp': datetime.utcnow().isoformat() + 'Z',
        }

    def _save_snapshot_cb(save_path):
        torch.save(model.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))

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
            random_prob=cfg.MLM_RANDOM_PROB, keep_prob=cfg.MLM_KEEP_PROB,
        )
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        try:
            with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            logits = outputs.get('mlm_logits', outputs.get('logits'))
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / accum_steps
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            if micro_batch_size > 4:
                micro_batch_size //= 2
                accum_steps = max(1, cfg.EFFECTIVE_BATCH_SIZE // micro_batch_size)
                logger.warning(f"OOM. Halved micro-batch size to {micro_batch_size}.")
                gen = _make_gen(doc_cursor[0])
                continue
            logger.error("OOM even at minimum micro-batch size!")
            raise

        # Count only real (non-padding) tokens toward the budget
        tokens_processed += int(attention_mask.sum().item())

        if (step + 1) % accum_steps == 0:
            # clip_grad_norm_ returns the pre-clip total norm. If that norm is
            # NaN/Inf, applying the update writes non-finite values into the
            # weights and the run is unrecoverable from that point on. Skipping
            # the update (without advancing the scheduler) is the standard
            # recovery; aborting after a sustained run of them turns a silent
            # 8-hour NaN burn into an immediate, diagnosable failure.
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            if not torch.isfinite(gnorm):
                nonfinite_grads += 1
                consecutive_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if consecutive_nonfinite <= 3 or consecutive_nonfinite % 100 == 0:
                    logger.warning(
                        f"NON-FINITE grad norm at optimizer step "
                        f"~{(step + 1) // accum_steps} (total skipped "
                        f"{nonfinite_grads}, consecutive {consecutive_nonfinite}) "
                        f"-- update SKIPPED")
                limit = getattr(cfg, "MAX_CONSECUTIVE_NONFINITE", 50)
                if consecutive_nonfinite >= limit:
                    logger.error(
                        f"ABORTING {run_id}: {consecutive_nonfinite} consecutive "
                        f"non-finite gradient norms -- weights are unrecoverable. "
                        f"Lower LEARNING_RATE_OVERRIDES[{arch!r}] and rerun.")
                    raise RuntimeError(
                        f"training diverged (non-finite gradients) in {run_id}")
                continue
            consecutive_nonfinite = 0
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            actual_step = (step + 1) // accum_steps

            # Adaptive validation cadence: when the last measured loss is
            # within VAL_FINE_MARGIN of the next uncrossed band, validate
            # every VAL_FINE_EVERY_STEPS so band overshoot stays bounded
            # (iso-loss matching is the paper's core control).
            fine_every = getattr(cfg, 'VAL_FINE_EVERY_STEPS', 500)
            fine_margin = getattr(cfg, 'VAL_FINE_MARGIN', 0.15)
            remaining_bands = iso_tracker.get_remaining_bands()
            near_band = (last_val_loss is not None and remaining_bands and
                         (last_val_loss - max(remaining_bands)) <= fine_margin)
            val_interval = fine_every if near_band else cfg.VAL_EVERY_STEPS

            if actual_step % val_interval == 0:
                val_loss, p_perp, mask_acc = evaluate(model, val_data, tokenizer, device)
                last_val_loss = val_loss
                elapsed = time.time() - start_time
                logger.info(f"Step {actual_step} | Tokens {tokens_processed / 1e6:.1f}M | "
                            f"Val Loss: {val_loss:.4f} | PP: {p_perp:.2f}")

                crossed_bands = iso_tracker.update(actual_step, val_loss, _save_snapshot_cb)
                for band in crossed_bands:
                    save_mlm_summary(_identity_row(val_loss, p_perp, mask_acc,
                                                   elapsed, band=band), summary_path)
                    _write_iso_index_row(summary_path, cfg, arch, size, seed,
                                         model_info, stream_count, merge_at,
                                         band, val_loss, actual_step,
                                         os.path.join(models_dir, 'iso_band_models', run_id))

            if token_markers and tokens_processed >= token_markers[0]:
                marker = token_markers.pop(0)
                marker_dir = os.path.join(models_dir, 'token_marker_models', run_id,
                                          f"tokens_{marker}")
                os.makedirs(marker_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(marker_dir, 'pytorch_model.bin'))
                val_loss, p_perp, mask_acc = evaluate(model, val_data, tokenizer, device)
                elapsed = time.time() - start_time
                save_mlm_summary(_identity_row(val_loss, p_perp, mask_acc,
                                               elapsed, marker=marker), summary_path)

            if actual_step % checkpoint_every == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                tmp_path = ckpt_path + '.tmp'
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iso_tracker': iso_tracker.state_dict(),
                    'token_markers': token_markers,
                    'step': step + 1,
                    'tokens_processed': tokens_processed,
                    'docs_consumed': doc_cursor[0],
                    'micro_batch_size': micro_batch_size,
                    'rng': _rng_state_dict(),
                }, tmp_path)
                os.replace(tmp_path, ckpt_path)
                logger.info(f"Checkpoint written at optimizer step {actual_step}.")

        step += 1

    logger.info(f"Finished {run_id}: {tokens_processed / 1e6:.1f}M tokens, "
                f"bands crossed: {iso_tracker.get_crossed_bands()}")

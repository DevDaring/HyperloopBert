"""
Calibration probe 2: why is MFU only 5.3%?

Hypotheses:
  (a) gradient checkpointing is ON for `base` (cfg 'auto') -- pure recompute
      overhead, pointless when only 5.4GB of 80GB is used;
  (b) micro-batch far too small for an H100 (per-step launch/sync overhead
      dominates -- evidenced by throughput scaling LINEARLY with batch);
  (c) per-micro-step .item() calls force a GPU sync every step.

Tests grad-ckpt ON vs OFF across micro-batch sizes, with accum=1 (so
EFFECTIVE_BATCH_SIZE == micro). NOTE: changing effective batch invalidates the
lr=1e-4 that was validated at batch 64 -- LR is re-validated separately before
any real run.
"""
import os, sys, time
sys.path.insert(0, os.path.abspath('.'))
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from common.architectures import build_model, apply_mlm_mask
from common.train_loop import memmap_generator, seed_everything
import Stage1.config_stage1 as cfg

device = 'cuda'
tok = PreTrainedTokenizerFast.from_pretrained('data/tokenizer')
BIN = 'data/fineweb-edu/train_128.bin'
FL_PER_TOK = 6 * (12 * (4*768**2 + 2*768*3072))   # base, non-embedding


def bench(micro, ckpt, seconds=55):
    seed_everything(42)
    model = build_model('VanillaBERT', 'base').to(device)
    model.train()
    if ckpt and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=cfg.ADAMW_BETAS,
                            eps=cfg.ADAMW_EPS, weight_decay=cfg.WEIGHT_DECAY)
    gen = memmap_generator(BIN, cfg.SEQ_LENGTH, micro)
    for _ in range(4):                      # warmup
        b = next(gen).to(device, non_blocking=True)
        m, l = apply_mlm_mask(b, tok, prob=cfg.MLM_PROBABILITY, mask_prob=cfg.MLM_MASK_PROB,
                              random_prob=cfg.MLM_RANDOM_PROB, keep_prob=cfg.MLM_KEEP_PROB)
        am = (b != tok.pad_token_id).long()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            o = model(input_ids=m, attention_mask=am)
        lg = o.get('mlm_logits', o.get('logits'))
        F.cross_entropy(lg.view(-1, lg.size(-1)).float(), l.view(-1), ignore_index=-100).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # accumulate token count on GPU; sync ONCE at the end (avoids per-step stalls)
    tok_count = torch.zeros((), dtype=torch.long, device=device)
    t0, steps = time.time(), 0
    while time.time() - t0 < seconds:
        b = next(gen).to(device, non_blocking=True)
        m, l = apply_mlm_mask(b, tok, prob=cfg.MLM_PROBABILITY, mask_prob=cfg.MLM_MASK_PROB,
                              random_prob=cfg.MLM_RANDOM_PROB, keep_prob=cfg.MLM_KEEP_PROB)
        am = (b != tok.pad_token_id).long()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            o = model(input_ids=m, attention_mask=am)
        lg = o.get('mlm_logits', o.get('logits'))
        loss = F.cross_entropy(lg.view(-1, lg.size(-1)).float(), l.view(-1), ignore_index=-100)
        loss.backward()
        tok_count += am.sum()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step(); opt.zero_grad(set_to_none=True)
        steps += 1
    torch.cuda.synchronize()
    el = time.time() - t0
    tps = int(tok_count.item()) / el
    mem = torch.cuda.max_memory_allocated()/1e9
    tfl = tps * FL_PER_TOK / 1e12
    print(f"  micro={micro:<5} ckpt={'ON ' if ckpt else 'OFF'}  {tps:>10,.0f} tok/s  "
          f"{tfl:6.1f} TFLOPS  MFU={tfl/990*100:5.1f}%  mem={mem:5.1f}GB  steps={steps}",
          flush=True)
    del model, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return tps

print("="*96)
print("PROBE 2 -- gradient checkpointing and micro-batch scaling (accum=1)")
print("="*96)
best = (0, None)
for ckpt in (True, False):
    for micro in (64, 128, 256, 512):
        try:
            t = bench(micro, ckpt)
            if t > best[0]:
                best = (t, (micro, ckpt))
        except torch.cuda.OutOfMemoryError:
            print(f"  micro={micro:<5} ckpt={'ON ' if ckpt else 'OFF'}  OOM")
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"  micro={micro:<5} ckpt={ckpt}  failed: {str(e)[:60]}")

tps, cfgbest = best
print("="*96)
print(f"BEST: micro={cfgbest[0]} ckpt={'ON' if cfgbest[1] else 'OFF'} -> {tps:,.0f} tok/s "
      f"({tps/103809:.1f}x the micro=64+ckpt baseline)")
for n_runs, per in ((8, 8e9), (8, 6e9), (8, 4e9), (4, 8e9)):
    print(f"  {n_runs} runs x {per/1e9:.0f}B tokens = {n_runs*per/tps/3600:6.1f} h")
print(f"  in 32h of training: {tps*32*3600/1e9:.1f}B total -> "
      f"{tps*32*3600/8/1e9:.2f}B per run across 8 runs")
print("="*96)

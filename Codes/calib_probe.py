"""
H100 calibration probe. Measures what the schedule must be sized from:
  1. real tokens/sec at `base` with the pre-tokenized memmap loader
  2. best micro-batch size (H100 has 80GB; the L4-era micro=16 under-fills it)
  3. early loss trajectory -> used to derive iso-loss bands FROM DATA

EFFECTIVE_BATCH_SIZE stays 64 so the validated lr=1e-4 remains correct; only
the micro-batch (i.e. gradient-accumulation granularity) is varied, which
changes speed but NOT the optimizer trajectory.
"""
import os, sys, time, math
sys.path.insert(0, os.path.abspath('.'))
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from common.architectures import build_model, apply_mlm_mask
from common.train_loop import memmap_generator, seed_everything, get_lr_schedule
import Stage1.config_stage1 as cfg

device = 'cuda'
tok = PreTrainedTokenizerFast.from_pretrained('data/tokenizer')
BIN = 'data/fineweb-edu/train_128.bin'
EFF = 64
LR = 1e-4

def throughput(micro, seconds=70):
    seed_everything(42)
    model = build_model('VanillaBERT', 'base').to(device)
    model.train()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    accum = max(1, EFF // micro)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=cfg.ADAMW_BETAS,
                            eps=cfg.ADAMW_EPS, weight_decay=cfg.WEIGHT_DECAY)
    gen = memmap_generator(BIN, cfg.SEQ_LENGTH, micro)
    # warmup
    for _ in range(6):
        b = next(gen).to(device)
        m, l = apply_mlm_mask(b, tok, prob=cfg.MLM_PROBABILITY, mask_prob=cfg.MLM_MASK_PROB,
                             random_prob=cfg.MLM_RANDOM_PROB, keep_prob=cfg.MLM_KEEP_PROB)
        am = (b != tok.pad_token_id).long()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            o = model(input_ids=m, attention_mask=am)
        lg = o.get('mlm_logits', o.get('logits'))
        loss = F.cross_entropy(lg.view(-1, lg.size(-1)).float(), l.view(-1), ignore_index=-100)
        (loss/accum).backward()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0, toks, n = time.time(), 0, 0
    while time.time() - t0 < seconds:
        b = next(gen).to(device)
        m, l = apply_mlm_mask(b, tok, prob=cfg.MLM_PROBABILITY, mask_prob=cfg.MLM_MASK_PROB,
                             random_prob=cfg.MLM_RANDOM_PROB, keep_prob=cfg.MLM_KEEP_PROB)
        am = (b != tok.pad_token_id).long()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            o = model(input_ids=m, attention_mask=am)
        lg = o.get('mlm_logits', o.get('logits'))
        loss = F.cross_entropy(lg.view(-1, lg.size(-1)).float(), l.view(-1), ignore_index=-100)
        (loss/accum).backward()
        toks += int(am.sum().item()); n += 1
        if n % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    el = time.time() - t0
    tps = toks/el
    mem = torch.cuda.max_memory_allocated()/1e9
    # non-embedding FLOPs/token for base: 12 layers x (4h^2 + 2*h*4h), h=768
    fl_per_tok = 6 * (12 * (4*768**2 + 2*768*3072))
    tflops = tps * fl_per_tok / 1e12
    print(f"  micro={micro:<4} {tps:>9,.0f} tok/s   {tflops:6.1f} TFLOPS  "
          f"MFU={tflops/990*100:5.1f}%  peakmem={mem:5.1f}GB", flush=True)
    del model, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return tps

print("="*78)
print("H100 CALIBRATION -- VanillaBERT base, pre-tokenized memmap, grad-ckpt ON")
print("EFFECTIVE_BATCH_SIZE=64 fixed (keeps validated lr=1e-4 valid)")
print("="*78)
best, best_m = 0, 16
for micro in (16, 32, 64):
    try:
        t = throughput(micro)
        if t > best: best, best_m = t, micro
    except torch.cuda.OutOfMemoryError:
        print(f"  micro={micro:<4} OOM"); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  micro={micro:<4} failed: {str(e)[:70]}")

print("="*78)
print(f"BEST: micro={best_m}  {best:,.0f} tok/s")
for label, hrs in (("per 8B-token run", 8e9/best/3600), ("8 runs x 8B", 8*8e9/best/3600)):
    print(f"  {label:22s}: {hrs:6.2f} h")
print(f"  tokens affordable in 34h of training: {best*34*3600/1e9:.1f}B total "
      f"-> {best*34*3600/8/1e9:.1f}B per run across 8 runs")
print("="*78)

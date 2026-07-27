"""
LR validation at the new effective batch (512), plus iso-loss band derivation.

Why: micro=512/accum=1 gives the best throughput but raises EFFECTIVE_BATCH_SIZE
from the validated 64 to 512. The earlier sweep showed this post-norm model
collapses at 5e-4 at BOTH batch 64 and batch 256, so linear LR scaling cannot be
assumed. Each candidate LR is run on the REAL schedule shape (warmup sized for
the full planned budget) and judged on:
    loss        -- must fall well below the ~7.2 unigram plateau
    ctx_cos     -- must fall (context actually being used); ~1.0 == collapse

The winner's trajectory is then used to derive achievable iso-loss bands, so the
bands come from measurement rather than assumption (the failure that wasted the
200M-token run).
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

MICRO = 512
EFF = 512
PLANNED_BUDGET = float(os.environ.get('PLANNED_BUDGET', '6e9'))   # full run size
TEST_TOKENS = float(os.environ.get('TEST_TOKENS', '250e6'))
CTXS = ["The capital of France is [MASK].",
        "He went to the [MASK] to buy milk.",
        "Water boils at one hundred [MASK]."]


def ctx_cos(model):
    model.eval(); ds = []
    for c in CTXS:
        e = tok(c.replace('[MASK]', tok.mask_token), return_tensors='pt').to(device)
        p = (e['input_ids'][0] == tok.mask_token_id).nonzero()[0].item()
        with torch.no_grad():
            o = model(input_ids=e['input_ids'], attention_mask=e['attention_mask'])
        lg = o.get('mlm_logits', o.get('logits'))
        ds.append(F.softmax(lg[0, p].float(), -1))
    model.train()
    v = [F.cosine_similarity(ds[i].unsqueeze(0), ds[j].unsqueeze(0)).item()
         for i in range(len(ds)) for j in range(i+1, len(ds))]
    return sum(v)/len(v)


def run(lr):
    seed_everything(42)
    model = build_model('VanillaBERT', 'base').to(device); model.train()
    total_steps = int(PLANNED_BUDGET / (EFF * cfg.SEQ_LENGTH * 0.90))
    warmup = int(total_steps * cfg.WARMUP_RATIO)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=cfg.ADAMW_BETAS,
                            eps=cfg.ADAMW_EPS, weight_decay=cfg.WEIGHT_DECAY)
    sched = get_lr_schedule(opt, warmup, total_steps)
    gen = memmap_generator(BIN, cfg.SEQ_LENGTH, MICRO)
    print(f"\n--- lr={lr:g}  (warmup {warmup} of {total_steps} steps for a "
          f"{PLANNED_BUDGET/1e9:.0f}B-token run) ---", flush=True)
    print(f"{'tokens_M':>9} {'loss':>7} {'ppl':>9} {'ctx_cos':>8} {'lr':>9}")
    tok_count = torch.zeros((), dtype=torch.long, device=device)
    seen, nxt, traj = 0, 50e6, []
    t0 = time.time()
    while seen < TEST_TOKENS:
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
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (seen := int(tok_count.item())) >= nxt:
            L = loss.item(); c = ctx_cos(model)
            traj.append((seen, L, c))
            print(f"{seen/1e6:9.1f} {L:7.3f} {math.exp(min(L,20)):9.1f} {c:8.4f} "
                  f"{sched.get_last_lr()[0]:9.2e}", flush=True)
            nxt += 50e6
    el = time.time() - t0
    print(f"   {seen/1e6:.0f}M tokens in {el:.0f}s = {seen/el:,.0f} tok/s")
    final = traj[-1] if traj else (0, 99, 1.0)
    del model, opt; torch.cuda.empty_cache()
    return {'lr': lr, 'loss': final[1], 'ctx_cos': final[2], 'traj': traj, 'tps': seen/el}

print("="*84)
print(f"LR VALIDATION at EFFECTIVE_BATCH_SIZE={EFF} (micro={MICRO}, ckpt OFF)")
print(f"each candidate: {TEST_TOKENS/1e6:.0f}M tokens on the real schedule shape")
print("="*84)

results = []
for lr in (1e-4, 3e-4, 6e-4):
    try:
        results.append(run(lr))
    except Exception as e:
        print(f"lr={lr:g} FAILED: {str(e)[:90]}")

print("\n" + "="*84)
print(f"{'lr':>8} {'final loss':>11} {'ctx_cos':>9}  verdict")
ok = []
for r in results:
    good = r['loss'] < 6.8 and r['ctx_cos'] < 0.95
    if good: ok.append(r)
    print(f"{r['lr']:8.1e} {r['loss']:11.3f} {r['ctx_cos']:9.4f}  "
          f"{'OK' if good else 'COLLAPSED/STUCK'}")

if ok:
    best = min(ok, key=lambda r: r['loss'])
    print(f"\nCHOSEN lr = {best['lr']:g}  (loss {best['loss']:.3f}, "
          f"ctx_cos {best['ctx_cos']:.4f}, {best['tps']:,.0f} tok/s)")
    # Extrapolate loss at the full budget: loss ~ a - b*ln(tokens)
    t = best['traj']
    if len(t) >= 2:
        (t1, l1, _), (t2, l2, _) = t[0], t[-1]
        b = (l1 - l2) / math.log(t2 / t1)
        pred = l2 - b * math.log(PLANNED_BUDGET / t2)
        print(f"extrapolated loss at {PLANNED_BUDGET/1e9:.0f}B tokens: ~{pred:.2f} "
              f"(ppl ~{math.exp(min(pred,20)):.0f})")
        # Bands must be crossed by EVERY arch, so keep them above the prediction.
        bands = [round(pred + d, 1) for d in (1.2, 0.9, 0.6, 0.35)]
        print(f"SUGGESTED iso-loss bands (all arms should cross): {bands}")
        print(f"SUGGESTED quality screen: pp <= {math.exp(min(pred+0.5,20)):.0f}")
else:
    print("\nNO LR PASSED -- do not start the main run; re-tune before spending budget.")
print("="*84)

"""
Short LR / batch sweep on the REAL corpus to find a config that actually learns.

Pass criterion: train loss must fall clearly below the unigram-entropy plateau
(~7.2 nats) AND the model must become context-sensitive (cosine between
different contexts well below 1.0). Each config sees the same token budget and
the same seed, so the comparison is fair.
"""
import os, sys, time, math
sys.path.insert(0, os.path.abspath('.'))
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from common.architectures import build_model, apply_mlm_mask
from common.train_loop import data_generator, seed_everything, get_lr_schedule
import Stage1.config_stage1 as cfg

device = 'cuda'
tok = PreTrainedTokenizerFast.from_pretrained('data/tokenizer')
TRAIN = 'data/fineweb-edu/train_filtered.jsonl'

TOKEN_BUDGET = 6_000_000          # per config
MICRO = 16
CTXS = ["The capital of France is [MASK].",
        "He went to the [MASK] to buy milk.",
        "Water boils at one hundred [MASK]."]


def context_sensitivity(model):
    model.eval()
    ds = []
    for c in CTXS:
        e = tok(c.replace('[MASK]', tok.mask_token), return_tensors='pt').to(device)
        p = (e['input_ids'][0] == tok.mask_token_id).nonzero()[0].item()
        with torch.no_grad():
            o = model(input_ids=e['input_ids'], attention_mask=e['attention_mask'])
        lg = o.get('mlm_logits', o.get('logits'))
        ds.append(F.softmax(lg[0, p].float(), -1))
    model.train()
    cos = [F.cosine_similarity(ds[i].unsqueeze(0), ds[j].unsqueeze(0)).item()
           for i in range(len(ds)) for j in range(i + 1, len(ds))]
    return sum(cos) / len(cos)


def trial(lr, eff_batch, label):
    seed_everything(42)
    model = build_model('VanillaBERT', 'tiny').to(device)
    model.train()
    accum = max(1, eff_batch // MICRO)
    tokens_per_step = eff_batch * cfg.SEQ_LENGTH * 0.90
    total_steps = int(TOKEN_BUDGET / tokens_per_step)
    warmup = max(10, int(total_steps * cfg.WARMUP_RATIO))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=cfg.ADAMW_BETAS,
                            eps=cfg.ADAMW_EPS, weight_decay=cfg.WEIGHT_DECAY)
    sched = get_lr_schedule(opt, warmup, total_steps)
    gen = data_generator(TRAIN, tok, cfg.SEQ_LENGTH, MICRO)

    tokens = 0
    micro = 0
    losses = []
    t0 = time.time()
    while tokens < TOKEN_BUDGET:
        try:
            batch = next(gen)
        except StopIteration:
            break
        ids = batch.to(device)
        m_ids, labels = apply_mlm_mask(ids, tok, prob=cfg.MLM_PROBABILITY,
                                       mask_prob=cfg.MLM_MASK_PROB,
                                       random_prob=cfg.MLM_RANDOM_PROB,
                                       keep_prob=cfg.MLM_KEEP_PROB)
        am = (ids != tok.pad_token_id).long()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(input_ids=m_ids, attention_mask=am)
        lg = out.get('mlm_logits', out.get('logits'))
        loss = F.cross_entropy(lg.view(-1, lg.size(-1)).float(), labels.view(-1),
                               ignore_index=-100)
        (loss / accum).backward()
        losses.append(loss.item())
        tokens += int(am.sum().item())
        micro += 1
        if micro % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

    first = sum(losses[:50]) / max(1, len(losses[:50]))
    last = sum(losses[-50:]) / max(1, len(losses[-50:]))
    cos = context_sensitivity(model)
    verdict = "LEARNING" if (last < 6.5 and cos < 0.99) else "COLLAPSED/STUCK"
    print(f"{label:34s} start={first:5.2f} end={last:5.2f} "
          f"drop={first-last:5.2f} ctx_cos={cos:.4f}  {verdict}   "
          f"[{time.time()-t0:.0f}s]")
    del model, opt
    torch.cuda.empty_cache()
    return last, cos


print("=" * 96)
print(f"LR/batch sweep -- {TOKEN_BUDGET/1e6:.0f}M tokens per config, VanillaBERT tiny, seed 42")
print("unigram plateau is ~7.2 ; PASS needs end-loss < 6.5 AND ctx_cos < 0.99")
print("=" * 96)

trial(5e-4, 64,  "A) lr=5e-4 batch=64  (CURRENT)")
trial(1e-4, 64,  "B) lr=1e-4 batch=64")
trial(3e-5, 64,  "C) lr=3e-5 batch=64")
trial(5e-4, 256, "D) lr=5e-4 batch=256 (accum x4)")
trial(2e-4, 256, "E) lr=2e-4 batch=256 (accum x4)")

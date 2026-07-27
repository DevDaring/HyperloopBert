"""
Confirmation run at lr=1e-4: does it escape the unigram plateau, and can it
plausibly reach the iso-loss bands (4.0 .. 3.1) within the 200M-token budget?

Logs val-style loss + context sensitivity every 2.5M tokens so the trajectory
(not just an endpoint) can be extrapolated.
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

LR = float(os.environ.get('SWEEP_LR', '1e-4'))
TOKEN_BUDGET = int(float(os.environ.get('SWEEP_TOKENS', '30e6')))
# Schedule is sized for the REAL 200M run so the LR curve matches what the
# actual Stage 1 run would see over this same token range.
REAL_BUDGET = cfg.MAX_TOKENS
MICRO = 16
EFF = 64
LOG_EVERY = 2_500_000

CTXS = ["The capital of France is [MASK].",
        "He went to the [MASK] to buy milk.",
        "Water boils at one hundred [MASK]."]


def ctx_cos(model):
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
    v = [F.cosine_similarity(ds[i].unsqueeze(0), ds[j].unsqueeze(0)).item()
         for i in range(len(ds)) for j in range(i + 1, len(ds))]
    return sum(v) / len(v)


seed_everything(42)
model = build_model('VanillaBERT', 'tiny').to(device)
model.train()
accum = EFF // MICRO
tokens_per_step = EFF * cfg.SEQ_LENGTH * 0.90
total_steps = int(REAL_BUDGET / tokens_per_step)
warmup = int(total_steps * cfg.WARMUP_RATIO)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=cfg.ADAMW_BETAS,
                        eps=cfg.ADAMW_EPS, weight_decay=cfg.WEIGHT_DECAY)
sched = get_lr_schedule(opt, warmup, total_steps)
gen = data_generator(TRAIN, tok, cfg.SEQ_LENGTH, MICRO)

print(f"lr={LR:g}  budget={TOKEN_BUDGET/1e6:.0f}M  (LR schedule sized for the real {REAL_BUDGET/1e6:.0f}M run)")
print(f"{'tokens_M':>9} {'loss':>7} {'ppl':>10} {'ctx_cos':>8}  {'lr':>9}")

tokens = 0
micro = 0
recent = []
next_log = LOG_EVERY
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
    recent.append(loss.item())
    tokens += int(am.sum().item())
    micro += 1
    if micro % accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if tokens >= next_log:
        L = sum(recent[-200:]) / len(recent[-200:])
        print(f"{tokens/1e6:9.1f} {L:7.3f} {math.exp(min(L,20)):10.1f} "
              f"{ctx_cos(model):8.4f}  {sched.get_last_lr()[0]:9.2e}", flush=True)
        next_log += LOG_EVERY

print(f"\nelapsed {time.time()-t0:.0f}s  ({tokens/1e6:.1f}M tokens, "
      f"{tokens/(time.time()-t0):.0f} tok/s)")
print("iso-loss bands needed: 4.0 / 3.7 / 3.4 / 3.1 ; quality screen needs loss <= 4.09")

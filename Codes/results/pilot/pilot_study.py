"""
PILOT STUDY -- go/no-go before committing the full Stage 1 GPU budget.

Trains the Stage 1 contrast pair (VanillaBERT vs LoopedBERT) at the corrected
LR and then answers the only three questions that decide whether the full run
can possibly produce a publishable result:

  Q1 MEASURABILITY  Does VanillaBERT reach a quality where bias is measurable
                    at all? (loss / pseudo-perplexity, context sensitivity)
  Q2 CAPABILITY     Does VanillaBERT exhibit STATISTICALLY DETECTABLE baseline
                    stereotype bias? (bootstrap CI on the item-level preference
                    rate must exclude 0.5 from above.) If a model never acquired
                    the stereotypes, no architecture can be shown to "reduce"
                    them -- the whole program is untestable at this scale.
  Q3 SIGNAL         Is there any directional Vanilla-vs-Looped effect, and does
                    it favour SCH? Reported with an item-level paired
                    permutation test.

Prints an explicit PROCEED / STOP recommendation. Honest by construction: a
null here is a real answer, not a failure to be explained away.
"""
import os, sys, time, math, json
sys.path.insert(0, os.path.abspath('.'))
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import PreTrainedTokenizerFast

from common.architectures import build_model, apply_mlm_mask
from common.train_loop import (data_generator, seed_everything, get_lr_schedule,
                               prepare_validation_set, evaluate)
from common.bias_metrics import score_bias_pair
from common.stats_engine import bootstrap_ci, item_level_paired_contrast
import Stage1.config_stage1 as cfg

device = 'cuda'
tok = PreTrainedTokenizerFast.from_pretrained('data/tokenizer')
TRAIN = 'data/fineweb-edu/train_filtered.jsonl'
VAL = 'data/fineweb-edu/validation.jsonl'
CROWS = 'data/datasets_eval/multicrows/crows_pair_english.csv'

LR = float(os.environ.get('PILOT_LR', '1e-4'))
BUDGET = int(float(os.environ.get('PILOT_TOKENS', '80e6')))
SIZE = os.environ.get('PILOT_SIZE', 'tiny')
N_PAIRS = int(os.environ.get('PILOT_PAIRS', '400'))
MICRO, EFF = 16, 64

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


def train_one(arch, val_data):
    seed_everything(42)
    model = build_model(arch, SIZE).to(device)
    model.train()
    accum = EFF // MICRO
    tps = EFF * cfg.SEQ_LENGTH * 0.90
    total_steps = int(BUDGET / tps)
    warmup = max(50, int(total_steps * cfg.WARMUP_RATIO))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=cfg.ADAMW_BETAS,
                            eps=cfg.ADAMW_EPS, weight_decay=cfg.WEIGHT_DECAY)
    sched = get_lr_schedule(opt, warmup, total_steps)
    gen = data_generator(TRAIN, tok, cfg.SEQ_LENGTH, MICRO)
    tokens, micro, t0, nxt = 0, 0, time.time(), 20_000_000
    while tokens < BUDGET:
        try:
            b = next(gen)
        except StopIteration:
            break
        ids = b.to(device)
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
        tokens += int(am.sum().item()); micro += 1
        if micro % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if tokens >= nxt:
            vl, pp, _ = evaluate(model, val_data, tok, device)
            print(f"   [{arch}] {tokens/1e6:5.1f}M  val_loss={vl:.3f} pp={pp:8.1f} "
                  f"ctx_cos={ctx_cos(model):.4f}", flush=True)
            nxt += 20_000_000
    vl, pp, macc = evaluate(model, val_data, tok, device)
    print(f"   [{arch}] FINAL {tokens/1e6:.1f}M  val_loss={vl:.4f}  pp={pp:.1f}  "
          f"mask_acc={macc:.4f}  ctx_cos={ctx_cos(model):.4f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    model.eval()
    return model, {'val_loss': vl, 'pp': pp, 'mask_acc': macc, 'ctx_cos': ctx_cos(model)}


def bias_eval(model, df):
    """Per-item Effect_Size + stereotype-preference for one model."""
    rows = []
    for idx, r in df.iterrows():
        s = score_bias_pair(model, tok, r['stereo'], r['anti'], device, compute_ss=False)
        if s['Effect_Size'] is None:
            continue
        rows.append({'Row_Index': idx, 'Effect_Size': s['Effect_Size'],
                     'Stereotype_Preferred': s['Stereotype_Preferred']})
    return pd.DataFrame(rows)


print("=" * 84)
print(f"PILOT STUDY  size={SIZE}  lr={LR:g}  budget={BUDGET/1e6:.0f}M tokens/arch  "
      f"pairs={N_PAIRS}")
print("=" * 84)

val_data = prepare_validation_set(VAL, tok, cfg.SEQ_LENGTH, 2000,
                                  mlm_probability=cfg.MLM_PROBABILITY,
                                  mask_prob=cfg.MLM_MASK_PROB,
                                  random_prob=cfg.MLM_RANDOM_PROB)
df = pd.read_csv(CROWS)
if len(df) > N_PAIRS:
    df = df.sample(n=N_PAIRS, random_state=42)

results, biases = {}, {}
for arch in ['VanillaBERT', 'LoopedBERT']:
    print(f"\n--- training {arch} ---", flush=True)
    model, stats = train_one(arch, val_data)
    results[arch] = stats
    print(f"--- scoring bias for {arch} ({len(df)} pairs, FP32 PLL) ---", flush=True)
    biases[arch] = bias_eval(model, df)
    del model; torch.cuda.empty_cache()

# ---- REFERENCE ANCHOR -------------------------------------------------------
# A fully-trained public BERT scored on the SAME pairs with the SAME scorer.
# This calibrates "is stereotype bias measurable at all with our metric" and
# shows what a model of adequate quality looks like, so a null on our weak
# models can be interpreted rather than hand-waved.
print("\n--- reference anchor: bert-base-uncased (fully trained) ---", flush=True)
ref_stats = None
try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    ref_tok = AutoTokenizer.from_pretrained('bert-base-uncased')
    ref_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased').to(device).eval()
    _saved_tok = tok
    tok = ref_tok                      # bias_eval closes over `tok`
    ref_bias = bias_eval(ref_model, df)
    tok = _saved_tok
    ref_items = ref_bias.groupby('Row_Index')['Stereotype_Preferred'].mean().tolist()
    r_rate, r_lo, r_hi = bootstrap_ci(ref_items)
    ref_stats = {'rate': r_rate, 'ci_low': r_lo, 'ci_high': r_hi,
                 'detectable': bool(r_lo > 0.5)}
    print(f"   bert-base-uncased preference = {r_rate:.4f} "
          f"95% CI [{r_lo:.4f}, {r_hi:.4f}]  "
          f"{'DETECTABLE' if r_lo > 0.5 else 'not detectable'}")
    del ref_model; torch.cuda.empty_cache()
except Exception as e:
    print(f"   reference anchor unavailable: {e}")

print("\n" + "=" * 84)
print("PILOT RESULTS")
print("=" * 84)

# ---- Q1 MEASURABILITY -------------------------------------------------------
print("\n[Q1] MEASURABILITY -- did the models reach a usable quality?")
for a, s in results.items():
    verdict = "OK" if s['ctx_cos'] < 0.95 else "DEGENERATE (context ignored)"
    print(f"   {a:14s} val_loss={s['val_loss']:.3f}  pp={s['pp']:8.1f}  "
          f"mask_acc={s['mask_acc']:.3f}  ctx_cos={s['ctx_cos']:.4f}   {verdict}")
print(f"   quality screen threshold: pp <= {cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD} "
      f"(loss <= {math.log(cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD):.2f})")
print(f"   iso-loss bands required : {cfg.DEFAULT_ISO_BANDS}")

# ---- Q2 CAPABILITY ----------------------------------------------------------
print("\n[Q2] CAPABILITY -- does VanillaBERT show DETECTABLE baseline bias?")
v = biases['VanillaBERT']
per_item = v.groupby('Row_Index')['Stereotype_Preferred'].mean().tolist()
rate, lo, hi = bootstrap_ci(per_item)
cap_pass = lo > 0.5
print(f"   Vanilla stereotype preference = {rate:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
      f"vs chance 0.50")
print(f"   capability gate: {'PASS' if cap_pass else 'FAIL'} "
      f"({'CI excludes 0.5' if cap_pass else 'CI includes 0.5 -- no measurable bias to reduce'})")

# ---- Q3 SIGNAL --------------------------------------------------------------
print("\n[Q3] SIGNAL -- Vanilla vs Looped")
l = biases['LoopedBERT']
m1 = v.set_index('Row_Index')['Effect_Size']
m2 = l.set_index('Row_Index')['Effect_Size']
common = m1.index.intersection(m2.index)
deltas = (m1.loc[common] - m2.loc[common]).tolist()
res = item_level_paired_contrast(deltas, alternative='greater')
p1 = v.set_index('Row_Index')['Stereotype_Preferred'].loc[common].mean()
p2 = l.set_index('Row_Index')['Stereotype_Preferred'].loc[common].mean()
print(f"   preference rate: Vanilla {p1:.4f}  vs  Looped {p2:.4f}   "
      f"(delta {p1-p2:+.4f})")
print(f"   item-level delta(Effect_Size) = {res['mean_delta']:+.4f} "
      f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]  p={res['p_value']:.4f}  "
      f"d={res['cohens_d']:+.3f}  n={res['n_items']}")
print(f"   NOTE: losses differ ({results['VanillaBERT']['val_loss']:.3f} vs "
      f"{results['LoopedBERT']['val_loss']:.3f}); the full run matches on loss, "
      f"the pilot matches on token budget.")

# ---- RECOMMENDATION ---------------------------------------------------------
print("\n" + "=" * 84)
degenerate = any(s['ctx_cos'] >= 0.95 for s in results.values())
if degenerate:
    rec = ("STOP -- models are degenerate (context ignored). No bias measurement is "
           "meaningful at this scale/budget. Full Stage 1 cannot succeed.")
elif not cap_pass:
    rec = ("STOP -- baseline bias is NOT statistically detectable on VanillaBERT. "
           "There is no bias to reduce, so no architecture can be shown to reduce it. "
           "The pre-registered capability gate would FAIL on the full run too.")
elif res['mean_delta'] < 0 and res['p_value'] > 0.5:
    rec = ("CAUTION -- direction is REVERSED (Looped shows MORE stereotype "
           "preference). SCH is not supported at this scale; a full run would "
           "likely yield a NO-GO. Consider reporting as a negative result.")
else:
    rec = ("PROCEED -- baseline bias is measurable and the contrast is computable. "
           "The full Stage 1 run can produce an interpretable verdict "
           "(GO or a publishable NO-GO).")
print("RECOMMENDATION: " + rec)
print("=" * 84)

if ref_stats:
    print(f"\nREFERENCE: a fully-trained BERT scores {ref_stats['rate']:.4f} "
          f"[{ref_stats['ci_low']:.4f}, {ref_stats['ci_high']:.4f}] on the same pairs "
          f"with the same scorer.")
    print(f"           Our Vanilla scores {rate:.4f} [{lo:.4f}, {hi:.4f}] "
          f"at val_loss {results['VanillaBERT']['val_loss']:.2f}.")
    print("           If the reference IS detectable and ours is NOT, the metric works "
          "and our models are simply too undertrained.")

json.dump({'results': results, 'reference_bert_base': ref_stats,
           'capability': {'rate': rate, 'ci_low': lo, 'ci_high': hi, 'pass': bool(cap_pass)},
           'signal': {k: (float(x) if isinstance(x, (int, float)) else x)
                      for k, x in res.items()},
           'pref_vanilla': float(p1), 'pref_looped': float(p2),
           'recommendation': rec},
          open(os.path.expanduser('~/pilot_results.json'), 'w'), indent=2, default=str)
print("saved -> ~/pilot_results.json")

"""Is the model actually learning, or stuck at the unigram baseline?"""
import os, sys, math
sys.path.insert(0, os.path.abspath('.'))
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from common.architectures import build_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = PreTrainedTokenizerFast.from_pretrained('data/tokenizer')

ckpt_path = 'checkpoints/stage1/VanillaBERT_tiny_seed42/latest.pt'
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print("checkpoint step:", ck['step'], "tokens:", f"{ck['tokens_processed']/1e6:.1f}M")

# Learning-rate actually in effect
sched = ck.get('scheduler', {})
print("scheduler last_epoch:", sched.get('last_epoch'))
opt = ck.get('optimizer', {})
try:
    lrs = [g['lr'] for g in opt['param_groups']]
    print("optimizer LR right now:", lrs[:2])
except Exception as e:
    print("could not read LR:", e)

model = build_model('VanillaBERT', 'tiny')
model.load_state_dict(ck['model'])
model.to(device).eval()

# --- Do the weights look trained (not collapsed / not exploded)? ---
with torch.no_grad():
    for name in ['embeddings.word_embeddings.weight', 'encoder.0.ffn_linear1.weight',
                 'encoder.11.ffn_linear1.weight']:
        p = dict(model.named_parameters()).get(name)
        if p is not None:
            print(f"{name:45s} mean={p.mean():+.5f} std={p.std():.5f} "
                  f"absmax={p.abs().max():.4f}")

# --- What does it actually predict? ---
probe = "The capital of France is [MASK]."
enc = tok(probe.replace('[MASK]', tok.mask_token), return_tensors='pt').to(device)
pos = (enc['input_ids'][0] == tok.mask_token_id).nonzero()[0].item()
with torch.no_grad():
    out = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
logits = out.get('mlm_logits', out.get('logits'))
probs = F.softmax(logits[0, pos].float(), -1)
top_p, top_i = probs.topk(10)
print("\ntop-10 at [MASK] in:", probe)
for p, i in zip(top_p, top_i):
    print(f"   {tok.convert_ids_to_tokens(int(i)):>15s}  {float(p):.4f}")

# --- Entropy of the prediction: high entropy == not committing == unigram-ish ---
ent = -(probs * probs.clamp_min(1e-12).log()).sum()
print(f"\nprediction entropy: {float(ent):.3f} nats (uniform over 30522 = {math.log(30522):.3f})")

# --- Is this just the unigram distribution? Compare across DIFFERENT contexts ---
ctxs = ["The capital of France is [MASK].",
        "He went to the [MASK] to buy milk.",
        "Water boils at one hundred [MASK]."]
dists = []
for c in ctxs:
    e = tok(c.replace('[MASK]', tok.mask_token), return_tensors='pt').to(device)
    p_ = (e['input_ids'][0] == tok.mask_token_id).nonzero()[0].item()
    with torch.no_grad():
        o = model(input_ids=e['input_ids'], attention_mask=e['attention_mask'])
    lg = o.get('mlm_logits', o.get('logits'))
    dists.append(F.softmax(lg[0, p_].float(), -1))
# If context is ignored, all distributions are ~identical (cosine ~1.0)
for a in range(len(dists)):
    for b in range(a + 1, len(dists)):
        cos = F.cosine_similarity(dists[a].unsqueeze(0), dists[b].unsqueeze(0)).item()
        print(f"cosine(ctx{a}, ctx{b}) = {cos:.4f}   (~1.00 => context IGNORED, unigram only)")

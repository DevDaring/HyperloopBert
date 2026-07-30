"""
Second-instrument replication: StereoSet (intrasentence) on the four
comparison-point checkpoints. Inference only, CPU, no new training.

Protocol is matched to the CrowS-Pairs scoring already used, verified against the
stored per-item files rather than assumed:

    PLL(s) = (1 / |I|) * sum_{i in I} log P(t_i | s_masked_at_i)
    effect = PLL(stereotype) - PLL(anti-stereotype)

i.e. each sentence is normalised by its own scoreable-token count and the effect
is the plain difference of those means. [CLS] and [SEP] are excluded. Scoring is
float32 throughout, as in the original run.

StereoSet gives three continuations per context, labelled
0 = anti-stereotype, 1 = stereotype, 2 = unrelated, so the same paired contrast
can be formed and the conventional StereoSet summary scores can also be reported:

    ss   percentage of items where the stereotype continuation outscores the
         anti-stereotype one
    lms  percentage of items where a meaningful continuation outscores the
         unrelated one (a language-modelling sanity check)
    icat lms * min(ss, 100 - ss) / 50
"""
import os, re, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = "/home/Debz/Research/HyperloopBert"
sys.path.insert(0, os.path.join(ROOT, "Codes"))
CKPT = "/tmp/ss_ckpt"
OUT = os.path.join(ROOT, "analysis/fire2026/out")
os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(max(1, os.cpu_count() - 2))
DEVICE = "cpu"
CHUNK = 64          # masked copies scored per forward pass

RUNS = {
    "Vanilla":   ("VanillaBERT",      "VanillaBERT_base_seed42",            {}),
    "Looped":    ("LoopedBERT",       "LoopedBERT_base_seed42",             {}),
    "ALBERT":    ("ALBERTLoopedBERT", "ALBERTLoopedBERT_base_seed42",       {}),
    "Hyperloop": ("HyperloopBERT",    "HyperloopBERT_base_seed42_streams4", {"num_streams": 4}),
}


def load_model(arch, run, kwargs):
    from common.architectures import build_model
    m = build_model(arch, "base", **kwargs)
    p = os.path.join(CKPT, "models/stage3/iso_band_models", run,
                     "band_2p20/pytorch_model.bin")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"    state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            print("      e.g.", missing[:3])
    m.eval().float()
    return m


def get_logits(out):
    if isinstance(out, dict):
        for k in ("mlm_logits", "logits", "prediction_logits"):
            if k in out:
                return out[k]
        raise KeyError(f"no logits in {list(out.keys())}")
    return out[0]


@torch.no_grad()
def pll(model, tok, sentence, max_length=128):
    """Mean log-probability per scoreable token; None if unscoreable."""
    enc = tok(sentence, return_tensors="pt", max_length=max_length, truncation=True)
    ids, am = enc["input_ids"], enc["attention_mask"]
    n = ids.size(1) - 2                      # exclude [CLS] and [SEP]
    if n <= 0:
        return None
    total = 0.0
    for start in range(0, n, CHUNK):
        pos = list(range(start, min(start + CHUNK, n)))
        mids = ids.repeat(len(pos), 1)
        for r, i in enumerate(pos):
            mids[r, i + 1] = tok.mask_token_id
        logits = get_logits(model(input_ids=mids, attention_mask=am.repeat(len(pos), 1)))
        lp = F.log_softmax(logits.float(), dim=-1)
        for r, i in enumerate(pos):
            total += lp[r, i + 1, ids[0, i + 1]].item()
    return total / n


def main():
    from transformers import PreTrainedTokenizerFast
    from datasets import load_dataset

    tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(CKPT, "tokenizer/tokenizer.json"),
        mask_token="[MASK]", cls_token="[CLS]", sep_token="[SEP]",
        pad_token="[PAD]", unk_token="[UNK]")

    ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
    items = []
    for ex in ds:
        s = ex["sentences"]
        by = {g: t for g, t in zip(s["gold_label"], s["sentence"])}
        if 0 in by and 1 in by:
            items.append(dict(id=ex["id"], bias_type=ex["bias_type"],
                              stereo=by[1], anti=by[0], unrelated=by.get(2)))
    print(f"scoring {len(items)} StereoSet intrasentence items on CPU "
          f"({torch.get_num_threads()} threads)\n")

    for short, (arch, run, kw) in RUNS.items():
        dest = os.path.join(OUT, f"stereoset_{short}.csv")
        if os.path.exists(dest):
            print(f"  {short}: already scored, skipping")
            continue
        print(f"  {short}: loading {arch}")
        model = load_model(arch, run, kw)
        rows, t0 = [], time.time()
        for k, it in enumerate(items):
            ps, pa = pll(model, tok, it["stereo"]), pll(model, tok, it["anti"])
            pu = pll(model, tok, it["unrelated"]) if it["unrelated"] else None
            rows.append(dict(id=it["id"], bias_type=it["bias_type"],
                             PLL_stereo=ps, PLL_anti=pa, PLL_unrelated=pu,
                             effect=None if (ps is None or pa is None) else ps - pa))
            if (k + 1) % 300 == 0:
                el = time.time() - t0
                print(f"    {k+1}/{len(items)}  {el:.0f}s elapsed, "
                      f"~{el/(k+1)*(len(items)-k-1):.0f}s left", flush=True)
        pd.DataFrame(rows).to_csv(dest, index=False)
        print(f"    wrote {dest}  ({time.time()-t0:.0f}s)\n")
        del model


if __name__ == "__main__":
    main()

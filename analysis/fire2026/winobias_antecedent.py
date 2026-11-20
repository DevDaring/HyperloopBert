"""
WinoBias antecedent test for masked language models (FIRE revision).

The masked-pronoun probe used earlier compares he against she at the pronoun.
Pro and anti WinoBias items differ only in that pronoun, so the masked input is
the same for both and the probe cannot say whether the model knows WHICH
occupation the pronoun refers to.

This test asks that question directly, following the masked-LM substitution
method of Kocijan et al. (2019, ACL), "A Surprisingly Robust Trick for the
Winograd Schema Challenge":

  * the pronoun is replaced by each candidate antecedent ("the janitor",
    "the accountant"), with the candidate's tokens masked;
  * the masked LM scores the candidate's tokens at those positions
    (mean log-probability over the candidate's tokens);
  * the item is correct when the gold antecedent scores higher.

Pro and anti lines of WinoBias type 1 share the gold antecedent and differ only
in pronoun gender, so after substitution they are one item. The masculine
version of each line is used, because "his" is always possessive while "her"
can be possessive or objective. A first-mention heuristic (always pick the
occupation named first) is reported as a reference rate.

Usage
  python winobias_antecedent.py --raw DIR --out OUT.csv \
      [--tokenizer DIR --ckpt ARCH=PATH ...] [--hf bert-base-uncased ...]
"""
import argparse
import csv
import math
import os
import re
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Codes"))

MASC = {"he": "subj", "him": "obj", "his": "poss"}


def load_items(raw_dir, wtype=1):
    occs = []
    for fn in ("male_occupations.txt", "female_occupations.txt"):
        with open(os.path.join(raw_dir, fn)) as f:
            occs += [l.strip() for l in f if l.strip()]
    occs = sorted(set(occs), key=len, reverse=True)

    def lines(fn):
        with open(os.path.join(raw_dir, fn)) as f:
            return [re.sub(r"^\d+\s+", "", l.strip()) for l in f if l.strip()]

    pro, anti = lines(f"pro_stereotyped_type{wtype}.txt.test"), lines(f"anti_stereotyped_type{wtype}.txt.test")
    assert len(pro) == len(anti)
    items, skipped = [], 0
    for i, (p, a) in enumerate(zip(pro, anti)):
        line = None
        for cand in (p, a):
            br = re.findall(r"\[([^\]]+)\]", cand)
            if len(br) == 2 and br[1].lower() in MASC:
                line = cand
        if line is None:
            skipped += 1
            continue
        gold_np, pron = re.findall(r"\[([^\]]+)\]", line)
        plain = line.replace("[", "").replace("]", "")
        gold_occ = re.sub(r"^(the|a|an)\s+", "", gold_np, flags=re.I).lower()
        # the two occupation mentions, in order, before the pronoun
        pre = plain.lower().split(" " + pron.lower() + " ")[0] if (" " + pron.lower() + " ") in plain.lower() else plain.lower()
        found = []
        for o in occs:
            for m in re.finditer(r"\b(the|a|an)\s+" + re.escape(o) + r"\b", pre):
                found.append((m.start(), o))
        found = sorted(set(found))
        names = []
        for _, o in found:
            if o not in names:
                names.append(o)
        if gold_occ not in names or len(names) != 2:
            skipped += 1
            continue
        other = [o for o in names if o != gold_occ][0]
        # locate the bracketed pronoun in the plain sentence (after the first mention)
        m = re.search(r"\[" + re.escape(pron) + r"\]", line)
        start = len(line[:m.start()].replace("[", "").replace("]", ""))
        items.append(dict(idx=i, sentence=plain, pron=pron, pron_start=start,
                          pron_end=start + len(pron), gold=gold_occ, other=other,
                          first_mention=names[0]))
    return items, skipped


def candidate_text(occ, case):
    return f"the {occ}'s" if case == "poss" else f"the {occ}"


@torch.no_grad()
def score_candidate(model_fn, tok, item, occ):
    case = MASC[item["pron"].lower()]
    cand = candidate_text(occ, case)
    s = item["sentence"]
    left, right = s[:item["pron_start"]], s[item["pron_end"]:]
    if item["pron_start"] == 0:
        cand = cand[0].upper() + cand[1:]
    cand_ids = tok(cand, add_special_tokens=False)["input_ids"]
    left_ids = tok(left.rstrip(), add_special_tokens=False)["input_ids"] if left.strip() else []
    right_ids = tok(right.lstrip(), add_special_tokens=False)["input_ids"] if right.strip() else []
    ids = [tok.cls_token_id] + left_ids + [tok.mask_token_id] * len(cand_ids) + right_ids + [tok.sep_token_id]
    pos = list(range(1 + len(left_ids), 1 + len(left_ids) + len(cand_ids)))
    x = torch.tensor([ids])
    logits = model_fn(x)
    lp = F.log_softmax(logits[0].float(), dim=-1)
    return sum(lp[p, t].item() for p, t in zip(pos, cand_ids)) / len(cand_ids)


def run_model(name, model_fn, tok, items, writer):
    correct = 0
    for it in items:
        sg = score_candidate(model_fn, tok, it, it["gold"])
        so = score_candidate(model_fn, tok, it, it["other"])
        ok = int(sg > so)
        correct += ok
        writer.writerow(dict(model=name, idx=it["idx"], gold=it["gold"], other=it["other"],
                             score_gold=sg, score_other=so, correct=ok))
    n = len(items)
    acc = correct / n
    # Wilson 95% interval
    z = 1.96
    c = (acc + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    print(f"{name:40s} accuracy {acc:.3f}  ({correct}/{n})  95% CI [{c - h:.3f}, {c + h:.3f}]", flush=True)
    return dict(model=name, accuracy=acc, correct=correct, n=n, ci_low=c - h, ci_high=c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--type", type=int, default=1, choices=[1, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--ckpt", nargs="*", default=[], help="ARCH=path/to/pytorch_model.bin[:label]")
    ap.add_argument("--hf", nargs="*", default=[])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    items, skipped = load_items(a.raw, a.type)
    if int(os.environ.get('EVAL_LIMIT', '0') or 0):   # smoke test only
        items = items[:int(os.environ['EVAL_LIMIT'])]
    fm = sum(it["first_mention"] == it["gold"] for it in items) / len(items)
    print(f"{len(items)} items ({skipped} skipped); first-mention heuristic accuracy {fm:.3f}")

    summ = [dict(model="first-mention heuristic", accuracy=fm, correct=None, n=len(items),
                 ci_low=None, ci_high=None)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "idx", "gold", "other",
                                          "score_gold", "score_other", "correct"])
        w.writeheader()
        for name in a.hf:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            tk = AutoTokenizer.from_pretrained(name)
            md = AutoModelForMaskedLM.from_pretrained(name).eval().to(a.device)
            fn = lambda x, md=md: md(input_ids=x.to(a.device)).logits
            summ.append(run_model(name, fn, tk, items, w))
            del md
        if a.ckpt:
            from transformers import PreTrainedTokenizerFast
            from common.architectures import build_model
            tok = PreTrainedTokenizerFast(
                tokenizer_file=os.path.join(a.tokenizer, "tokenizer.json"),
                mask_token="[MASK]", cls_token="[CLS]", sep_token="[SEP]",
                pad_token="[PAD]", unk_token="[UNK]")
            for spec in a.ckpt:
                arch, rest = spec.split("=", 1)
                path, _, label = rest.partition(":")
                kw = {"num_streams": 4} if "Hyperloop" in arch else {}
                md = build_model(arch, "base", **kw)
                md.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
                md = md.eval().float().to(a.device)

                def fn(x, md=md):
                    o = md(input_ids=x.to(a.device))
                    if isinstance(o, dict):
                        for k in ("mlm_logits", "logits", "prediction_logits"):
                            if k in o:
                                return o[k]
                    return o[0]
                summ.append(run_model(label or arch, fn, tok, items, w))
                del md
    with open(a.out.replace(".csv", "_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)


if __name__ == "__main__":
    main()

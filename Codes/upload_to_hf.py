"""
upload_to_hf.py -- publish trained models, adapters and results to the Hub.

Target: https://huggingface.co/Debk/HyperloopBERT  (override with --repo)

SECURITY CONTRACT
-----------------
The token is read at runtime from Codes/.env (HUGGINGFACE_TOKEN) or the
environment. It is NEVER printed, never written to a file, and never passed on
a command line. Nothing under data/ and no .env is ever uploaded.

WHAT GETS UPLOADED
------------------
  models/<stage>/iso_band_models/**/pytorch_model.bin   (+ token_marker_models)
  results/<stage>/**                                     (CSV/JSON/markdown)
  figures/<stage>/**
  data/tokenizer/**                                      (needed to load models)
  PRE_REGISTRATION_AMENDMENT.md, Codes/README.md
plus an auto-generated README.md model card.

The dry_run/ sandbox is ALWAYS excluded -- it is QA scaffolding, not results.

Usage:
    python3 upload_to_hf.py --stage stage1
    python3 upload_to_hf.py --stage stage1 --dry-run     # list, upload nothing
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common.logging_setup import setup_logging
from common.env_loader import env_loader

logger = setup_logging('upload_to_hf')

BASE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE, '..'))


def _token():
    tok = env_loader.get('HUGGINGFACE_TOKEN') or env_loader.get('hf_token')
    if not tok:
        raise SystemExit("HUGGINGFACE_TOKEN not found in Codes/.env or environment.")
    return tok


def _collect(stage):
    """(local_path, path_in_repo) pairs. dry_run/ is always skipped."""
    items = []

    def add_tree(local_dir, prefix, exts=None):
        if not os.path.isdir(local_dir):
            return
        for root, dirs, files in os.walk(local_dir):
            dirs[:] = [d for d in dirs if d != 'dry_run']
            if 'dry_run' in root.split(os.sep):
                continue
            for fn in files:
                if exts and not fn.endswith(tuple(exts)):
                    continue
                lp = os.path.join(root, fn)
                rel = os.path.relpath(lp, local_dir)
                items.append((lp, f"{prefix}/{rel}".replace(os.sep, '/')))

    add_tree(os.path.join(BASE, 'models', stage), f"models/{stage}", ['.bin', '.json'])
    add_tree(os.path.join(BASE, 'results', stage), f"results/{stage}",
             ['.csv', '.json', '.md'])
    add_tree(os.path.join(BASE, 'figures', stage), f"figures/{stage}",
             ['.png', '.pdf', '.svg'])
    add_tree(os.path.join(BASE, 'data', 'tokenizer'), "tokenizer")

    for doc, dest in ((os.path.join(REPO_ROOT, 'PRE_REGISTRATION_AMENDMENT.md'),
                       'PRE_REGISTRATION_AMENDMENT.md'),
                      (os.path.join(BASE, 'README.md'), 'CODE_README.md')):
        if os.path.exists(doc):
            items.append((doc, dest))
    return items


def _model_card(stage, items):
    n_models = sum(1 for _, r in items if r.endswith('.bin'))
    return f"""---
license: apache-2.0
language: [en]
tags: [bert, masked-lm, fairness, bias-evaluation, weight-sharing, looped-transformer]
---

# HyperloopBERT — {stage}

Research artifacts for a controlled study of **cross-layer weight sharing and
stereotype association at matched model quality** (the Stereotype Consolidation
Hypothesis, SCH).

Uploaded {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · {n_models} checkpoint(s).

## Contents
- `models/{stage}/iso_band_models/` — checkpoints snapshotted at matched
  validation-loss (iso-loss) bands, the primary comparison points
- `models/{stage}/token_marker_models/` — checkpoints at fixed token budgets
- `results/{stage}/` — MLM quality, per-item bias scores, capability-gate
  evidence, statistics, generated paper outline
- `tokenizer/` — the shared WordPiece tokenizer (required to load any checkpoint)
- `PRE_REGISTRATION_AMENDMENT.md` — every deviation from the pre-registered plan

## Construct scope (read before using the bias numbers)
The bias endpoint is a **correlation**-type measure in the descriptive /
normative / correlation taxonomy of Wang et al. (2025), *Fairness through
Difference Awareness* (arXiv:2502.01926). These models measure stereotype
**association**; a lower preference rate is **not** a claim of difference-aware
fairness.

## Intended use
Research and fairness auditing only. These are small models pretrained from
scratch on a limited token budget — they are **architecture probes, not
production encoders**. The evaluation datasets contain stereotypical content by
design.

## Citation
See `PRE_REGISTRATION_AMENDMENT.md` and `CODE_README.md` for the full reference list.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='stage1')
    ap.add_argument('--repo', default='Debk/HyperloopBERT')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    items = _collect(args.stage)
    if not items:
        logger.error(f"Nothing to upload for {args.stage} "
                     f"(no models/results found). Has the stage run?")
        return

    total_mb = sum(os.path.getsize(p) for p, _ in items) / 1e6
    logger.info(f"{len(items)} file(s), {total_mb:.1f} MB -> {args.repo}")
    for lp, rp in items[:25]:
        logger.info(f"   {rp}  ({os.path.getsize(lp)/1e6:.2f} MB)")
    if len(items) > 25:
        logger.info(f"   ... and {len(items)-25} more")

    if args.dry_run:
        logger.info("--dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=_token())          # token never logged
    api.create_repo(args.repo, repo_type='model', exist_ok=True, private=False)

    card = os.path.join(BASE, f'.hf_card_{args.stage}.md')
    with open(card, 'w', encoding='utf-8') as f:
        f.write(_model_card(args.stage, items))
    items.append((card, 'README.md'))

    ok = 0
    for lp, rp in items:
        try:
            api.upload_file(path_or_fileobj=lp, path_in_repo=rp,
                            repo_id=args.repo, repo_type='model')
            ok += 1
        except Exception as e:
            logger.error(f"failed {rp}: {str(e)[:160]}")
    os.remove(card)
    logger.info(f"uploaded {ok}/{len(items)} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()

"""
Dry_Run/dry_run_dataset.py

Dataset-stage dry run (spec section 11): verifies the environment, HF access,
a tiny slice of each evaluation dataset, tokenizer round-trip, and the
integrity suite on a small synthetic slice. Writes a machine-readable report
to Dry_Run/dry_run_report.json. Exits fast; downloads at most a few rows.
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.env_loader import env_loader

logger = setup_logging('dry_run_dataset')

REPORT_PATH = os.path.join(os.path.dirname(__file__), 'dry_run_report.json')


def check_env():
    """Report per-provider key counts (never the values)."""
    result = {}
    for provider in ['hf', 'gemini', 'deepseek', 'mistral', 'openrouter']:
        try:
            available = env_loader.is_provider_available(provider)
        except Exception:
            available = False
        result[provider] = bool(available)
        logger.info(f"Provider '{provider}' available: {available}")
    return {'status': 'PASS', 'providers': result}


def check_hf_slices():
    """Download a tiny slice of each dataset the pipeline uses."""
    from datasets import load_dataset
    checks = {}

    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        row = next(iter(ds))
        checks['fineweb_edu'] = 'PASS' if row.get('text') else 'FAIL (empty text)'
    except Exception as e:
        checks['fineweb_edu'] = f'FAIL ({e})'

    try:
        # Canonical public English CrowS-Pairs (Nangia et al. 2020). The old
        # 'HuggingFaceM4/Multi-lingual-crows-pairs' mirror no longer exists.
        ds = load_dataset("nyu-mll/crows_pairs", split="test")
        checks['multicrows'] = 'PASS' if len(ds) > 0 else 'FAIL (empty)'
    except Exception as e:
        checks['multicrows'] = f'FAIL ({e})'

    try:
        ds = load_dataset("wino_bias", "type1_pro", split="test",
                          trust_remote_code=True)
        checks['winobias'] = 'PASS' if len(ds) > 0 else 'FAIL (empty)'
    except Exception as e:
        checks['winobias'] = f'FAIL ({e})'

    status = 'PASS' if all(v == 'PASS' for v in checks.values()) else 'FAIL'
    for name, v in checks.items():
        logger.info(f"HF slice {name}: {v}")
    return {'status': status, 'checks': checks}


def check_integrity_smoke():
    """Run the corruption/dedup logic on a tiny synthetic JSONL slice."""
    import tempfile
    try:
        from common.integrity import run_integrity_suite
    except Exception as e:
        return {'status': f'FAIL (cannot import integrity: {e})'}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            slice_path = os.path.join(tmp, 'slice.jsonl')
            with open(slice_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps({'text': 'A valid document about mathematics.'}) + '\n')
                f.write(json.dumps({'text': 'A valid document about mathematics.'}) + '\n')  # duplicate
                f.write('{"text": "truncated\n')  # corrupt
                f.write(json.dumps({'text': 'Another valid document.'}) + '\n')
            quarantine = os.path.join(tmp, 'quarantine')
            os.makedirs(quarantine, exist_ok=True)
            summary = run_integrity_suite(tmp, os.path.join(tmp, 'manifest.json'),
                                          logger=logger, quarantine_dir=quarantine)
            ok = (summary.get('total_quarantined', 0) >= 1 and
                  summary.get('total_duplicates_removed', 0) >= 1)
            return {'status': 'PASS' if ok else 'FAIL (dedup/corruption not detected)',
                    'summary': {k: v for k, v in summary.items() if isinstance(v, (int, str))}}
    except Exception as e:
        return {'status': f'FAIL ({e})'}


def check_tokenizer_roundtrip():
    """If the shared tokenizer exists, verify special tokens and a round-trip."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tok_dir = os.path.join(base_dir, 'data', 'tokenizer')
    if not os.path.exists(tok_dir):
        return {'status': 'SKIP (tokenizer not trained yet)'}
    try:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(tok_dir)
        ids = tok("The quick brown fox.")['input_ids']
        ok = (tok.pad_token_id == 0 and tok.mask_token_id is not None and len(ids) > 2)
        return {'status': 'PASS' if ok else 'FAIL (unexpected special token ids)',
                'pad_id': tok.pad_token_id, 'mask_id': tok.mask_token_id}
    except Exception as e:
        return {'status': f'FAIL ({e})'}


def main():
    report = {
        'stage': 'dataset',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'junctions': {},
    }
    report['junctions']['env'] = check_env()
    report['junctions']['hf_slices'] = check_hf_slices()
    report['junctions']['integrity_smoke'] = check_integrity_smoke()
    report['junctions']['tokenizer'] = check_tokenizer_roundtrip()

    statuses = [j.get('status', 'FAIL') for j in report['junctions'].values()]
    report['overall'] = 'PASS' if all(s.startswith(('PASS', 'SKIP')) for s in statuses) else 'FAIL'

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Dataset dry run overall: {report['overall']} (report: {REPORT_PATH})")

    if report['overall'] != 'PASS':
        sys.exit(1)


if __name__ == "__main__":
    main()

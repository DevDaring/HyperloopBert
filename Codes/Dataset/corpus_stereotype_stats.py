"""
Dataset/corpus_stereotype_stats.py -- corpus-level stereotype statistics.

Motivation (pre-registration amendment A.2.3): FineWeb-Edu is heavily
filtered. If the stereotype associations the bias benchmarks test barely
occur in the training corpus, a null bias result means "never learned", not
"architecturally mitigated" -- and a near-chance baseline is expected, not
suspicious. This script produces the sanity table that separates the two
readings, and is the pre-registered defense if baseline bias lands near
chance.

Method (cheap, one corpus pass):
  For every benchmark pair (stereo, anti):
    - STEREO/ANTI terms: the words the manipulation changed (set difference
      of the two sentences' word sets).
    - CONTEXT terms: informative shared words (stopword-filtered), i.e. the
      attribute content the pair attaches to the demographic terms.
  Then stream the training corpus once, recording per-term document
  presence in a packed bit matrix, and report per pair:
    - document frequency of stereo/anti/context terms
    - documents where a stereo term CO-OCCURS with a context term
      (and the same for anti terms).

Outputs:
  data/corpus_stats/pair_cooccurrence.csv     (CORPUS_PAIR_STATS_COLUMNS)
  data/corpus_stats/category_summary.csv      (CORPUS_CATEGORY_STATS_COLUMNS)

Document sampling: --max-docs (default 200,000) bounds the pass; counts are
then estimates over the sampled prefix and Sampled_Docs records the
denominator. Use --max-docs 0 for the full corpus.
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.io_schemas import (
    CORPUS_PAIR_STATS_COLUMNS, CORPUS_CATEGORY_STATS_COLUMNS,
)

logger = setup_logging('corpus_stereotype_stats')

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

_WORD_RE = re.compile(r"[a-z][a-z']+")

# Minimal stopword list: function words that would make "context" vacuous.
_STOPWORDS = frozenset("""
a an the and or but if then than so as of at by for with about against
between into through during before after above below to from up down in out
on off over under again further once here there all any both each few more
most other some such no nor not only own same too very can will just don
should now is are was were be been being have has had having do does did
doing would could ought i'm you're he's she's it's we're they're i've you've
we've they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll
we'll they'll isn't aren't wasn't weren't hasn't haven't hadn't doesn't
don't didn't won't wouldn't shan't shouldn't can't cannot couldn't mustn't
let's that's who's what's here's there's when's where's why's how's because
what which who whom this that these those am it its they them their theirs
he she him her his hers you your yours we us our ours my mine me i was
""".split())

# Cap on context terms per pair (longest words first: a cheap proxy for
# informativeness that needs no corpus-wide document frequencies up front).
_MAX_CONTEXT_TERMS = 6


def _words(text):
    if not isinstance(text, str):
        return set()
    return set(_WORD_RE.findall(text.lower()))


def load_benchmark_pairs(eval_dir):
    """Yield (dataset, row_index, category, stereo_terms, anti_terms, context_terms)."""
    sources = {
        'multicrows': os.path.join(eval_dir, 'multicrows', 'crows_pair_english.csv'),
        'indian_bias': os.path.join(eval_dir, 'indian_bias', 'indian_bias_english.csv'),
    }
    pairs = []
    for ds_name, path in sources.items():
        if not os.path.exists(path):
            logger.warning(f"{ds_name}: {path} not found; skipped.")
            continue
        df = pd.read_csv(path)
        if 'stereo' not in df.columns or 'anti' not in df.columns:
            logger.warning(f"{ds_name}: no stereo/anti columns; skipped.")
            continue
        for idx, row in df.iterrows():
            sw, aw = _words(row['stereo']), _words(row['anti'])
            stereo_terms = {w for w in (sw - aw) if w not in _STOPWORDS}
            anti_terms = {w for w in (aw - sw) if w not in _STOPWORDS}
            context = sorted((sw & aw) - _STOPWORDS, key=len, reverse=True)
            context_terms = set(context[:_MAX_CONTEXT_TERMS])
            if not stereo_terms or not context_terms:
                continue
            category = row.get('bias_type', row.get('category', 'unknown'))
            pairs.append((ds_name, idx, category, stereo_terms, anti_terms,
                          context_terms))
    return pairs


def scan_corpus(corpus_path, term_to_idx, max_docs):
    """
    One pass over the corpus. Returns (bit_matrix, n_docs): bit_matrix has one
    packed-bit row per term marking the documents that contain it.
    """
    n_terms = len(term_to_idx)
    capacity = max_docs if max_docs > 0 else 1_000_000
    mat = np.zeros((n_terms, (capacity + 7) // 8), dtype=np.uint8)

    n_docs = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_docs > 0 and n_docs >= max_docs:
                break
            try:
                text = json.loads(line).get('text', '')
            except json.JSONDecodeError:
                continue
            if not text:
                continue
            if n_docs >= mat.shape[1] * 8:  # grow for --max-docs 0
                mat = np.concatenate(
                    [mat, np.zeros((n_terms, mat.shape[1]), dtype=np.uint8)], axis=1)
            byte_idx, bit = n_docs >> 3, 1 << (n_docs & 7)
            for w in _words(text):
                t = term_to_idx.get(w)
                if t is not None:
                    mat[t, byte_idx] |= bit
            n_docs += 1
            if n_docs % 50_000 == 0:
                logger.info(f"...scanned {n_docs} documents")
    return mat, n_docs


_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint32)


def _count(row_bits):
    return int(_POPCOUNT[row_bits].sum())


def _any_rows(mat, term_indices):
    if not term_indices:
        return np.zeros(mat.shape[1], dtype=np.uint8)
    return np.bitwise_or.reduce(mat[term_indices], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-docs', type=int, default=200_000,
                        help='Documents sampled from the corpus head '
                             '(0 = full corpus)')
    args = parser.parse_args()

    eval_dir = os.path.join(DATA_DIR, 'datasets_eval')
    out_dir = os.path.join(DATA_DIR, 'corpus_stats')

    corpus_path = os.path.join(DATA_DIR, 'fineweb-edu', 'train_filtered.jsonl')
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(DATA_DIR, 'fineweb-edu', 'train_raw.jsonl')
    if not os.path.exists(corpus_path):
        raise SystemExit(f"No training corpus found under "
                         f"{os.path.join(DATA_DIR, 'fineweb-edu')}. "
                         f"Run the Dataset stage first.")

    pairs = load_benchmark_pairs(eval_dir)
    if not pairs:
        raise SystemExit("No benchmark pairs loaded; run "
                         "Dataset/download_eval_datasets.py first.")
    logger.info(f"Loaded {len(pairs)} benchmark pairs.")

    vocab = sorted(set().union(*[s | a | c for _, _, _, s, a, c in pairs]))
    term_to_idx = {w: i for i, w in enumerate(vocab)}
    logger.info(f"Tracking {len(vocab)} distinct benchmark lexemes in "
                f"{corpus_path} (max_docs={args.max_docs}).")

    mat, n_docs = scan_corpus(corpus_path, term_to_idx, args.max_docs)
    logger.info(f"Scanned {n_docs} documents.")
    if args.max_docs > 0:
        logger.info("NOTE: counts are estimates over a sampled corpus prefix; "
                    "Sampled_Docs records the denominator. Use --max-docs 0 "
                    "for exact full-corpus counts.")

    timestamp = datetime.utcnow().isoformat() + 'Z'
    pair_rows = []
    for ds_name, idx, category, stereo_terms, anti_terms, context_terms in pairs:
        s_idx = [term_to_idx[w] for w in stereo_terms]
        a_idx = [term_to_idx[w] for w in anti_terms]
        c_idx = [term_to_idx[w] for w in context_terms]
        s_bits = _any_rows(mat, s_idx)
        a_bits = _any_rows(mat, a_idx)
        c_bits = _any_rows(mat, c_idx)
        pair_rows.append({
            'Dataset': ds_name,
            'Row_Index': idx,
            'Category': category,
            'Stereo_Terms': ' '.join(sorted(stereo_terms)),
            'Anti_Terms': ' '.join(sorted(anti_terms)),
            'Context_Terms': ' '.join(sorted(context_terms)),
            'Docs_With_Stereo_Term': _count(s_bits),
            'Docs_With_Anti_Term': _count(a_bits),
            'Docs_With_Context_Term': _count(c_bits),
            'Stereo_Cooccurrence_Docs': _count(s_bits & c_bits),
            'Anti_Cooccurrence_Docs': _count(a_bits & c_bits),
            'Sampled_Docs': n_docs,
            'Timestamp': timestamp,
        })

    os.makedirs(out_dir, exist_ok=True)
    pair_df = pd.DataFrame(pair_rows)[CORPUS_PAIR_STATS_COLUMNS]
    pair_path = os.path.join(out_dir, 'pair_cooccurrence.csv')
    pair_df.to_csv(pair_path, index=False)
    logger.info(f"Wrote {len(pair_df)} pair rows to {pair_path}")

    cat_rows = []
    for (ds_name, category), grp in pair_df.groupby(['Dataset', 'Category']):
        cat_rows.append({
            'Dataset': ds_name,
            'Category': category,
            'Pair_Count': int(len(grp)),
            'Mean_Stereo_Cooccurrence_Docs': float(grp['Stereo_Cooccurrence_Docs'].mean()),
            'Median_Stereo_Cooccurrence_Docs': float(grp['Stereo_Cooccurrence_Docs'].median()),
            'Mean_Anti_Cooccurrence_Docs': float(grp['Anti_Cooccurrence_Docs'].mean()),
            'Zero_Stereo_Cooccurrence_Fraction':
                float((grp['Stereo_Cooccurrence_Docs'] == 0).mean()),
            'Sampled_Docs': n_docs,
            'Timestamp': timestamp,
        })
    cat_df = pd.DataFrame(cat_rows)[CORPUS_CATEGORY_STATS_COLUMNS]
    cat_path = os.path.join(out_dir, 'category_summary.csv')
    cat_df.to_csv(cat_path, index=False)
    logger.info(f"Wrote {len(cat_df)} category rows to {cat_path}")

    zero_frac = float((pair_df['Stereo_Cooccurrence_Docs'] == 0).mean())
    logger.info(f"Overall: {zero_frac * 100:.1f}% of pairs have ZERO "
                f"stereo-term/context co-occurrence in the sampled corpus. "
                f"Report this table alongside any near-chance baseline.")


if __name__ == "__main__":
    main()

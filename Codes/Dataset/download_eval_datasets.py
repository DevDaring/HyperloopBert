import os
import sys
import json
import argparse
import logging
import pandas as pd
from datasets import load_dataset

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.env_loader import env_loader

logger = setup_logging('download_eval_datasets')

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
EVAL_DIR = os.path.join(DATA_DIR, 'datasets_eval')

# CITATION: Nangia, N. et al. (2020). CrowS-Pairs. EMNLP 2020.
# CITATION: Khandelwal, K. et al. (2023). Indian-BhED. arXiv:2309.08573.
# CITATION: Zhao, J. et al. (2018). WinoBias. NAACL 2018.
# DATASET WARNING: These datasets contain stereotypical content by design.
#           Research/fairness-audit use only.

_GENDERED_PRONOUNS = {'he', 'she', 'him', 'her', 'his', 'hers'}


def _standardize_pair_columns(df, dataset_label):
    """
    Rename sentence-pair columns to the canonical 'stereo'/'anti' names and
    FAIL LOUDLY if no known pair-column convention is found. A silent skip
    here would crash every downstream eval with an opaque KeyError.
    """
    rename_candidates = [
        ('sent_more', 'sent_less'),
        ('stereo_sentence', 'anti_sentence'),
        ('stereotype', 'anti_stereotype'),
        ('sentence_stereotypical', 'sentence_antistereotypical'),
    ]
    if 'stereo' in df.columns and 'anti' in df.columns:
        return df
    for more_col, less_col in rename_candidates:
        if more_col in df.columns and less_col in df.columns:
            return df.rename(columns={more_col: 'stereo', less_col: 'anti'})
    raise ValueError(
        f"{dataset_label}: could not find a sentence-pair column convention in "
        f"{list(df.columns)}. Expected one of {rename_candidates} or 'stereo'/'anti'."
    )


def download_multi_crows_pairs(namespace: str = 'Debk'):
    """Download Multi-CrowS-Pairs (English subset).

    Tries the project namespace first (spec 7.2: --dataset-namespace lets an
    anonymised mirror be substituted for blind submission), then falls back
    to the public multilingual CrowS-Pairs mirror.
    """
    out_dir = os.path.join(EVAL_DIR, 'multicrows')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'crows_pair_english.csv')

    if os.path.exists(out_path):
        logger.info(f"Multi-CrowS-Pairs already exists at {out_path}")
        return

    df = None
    for repo, kwargs in [(f"{namespace}/Multi-CrowS-Pairs", {}),
                         ("HuggingFaceM4/Multi-lingual-crows-pairs", {'name': 'english'})]:
        try:
            logger.info(f"Downloading Multi-CrowS-Pairs from {repo}...")
            dataset = load_dataset(repo, split="test", **kwargs)
            df = dataset.to_pandas()
            break
        except Exception as e:
            logger.warning(f"Could not load {repo}: {e}")

    if df is None:
        raise RuntimeError("Multi-CrowS-Pairs could not be downloaded from any source.")

    df = _standardize_pair_columns(df, 'Multi-CrowS-Pairs')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} pairs to {out_path}")


# Spec 7.2 category -> per-category filename (datasets_eval/indian_bias/english/).
# Canonical Indian-BhED + multilingual-extension categories.
_INDIAN_CATEGORY_FILES = {
    'Caste': 'Caste.csv',
    'Gender': 'Gender.csv',
    'Religion': 'India_Religious.csv',
    'Race_Ethnicity': 'Race.csv',
}

# Map raw category strings found in the source to the 4 canonical categories.
_INDIAN_CATEGORY_ALIASES = {
    'caste': 'Caste',
    'gender': 'Gender',
    'religion': 'Religion',
    'india_religious': 'Religion',
    'religious': 'Religion',
    'race': 'Race_Ethnicity',
    'race_ethnicity': 'Race_Ethnicity',
    'race_color': 'Race_Ethnicity',
    'ethnicity': 'Race_Ethnicity',
}


def _canonical_indian_category(raw) -> str:
    if not isinstance(raw, str):
        return 'unknown'
    return _INDIAN_CATEGORY_ALIASES.get(raw.strip().lower(), raw.strip())


def _write_indian_provenance(prov_dir, base_repo, n_pairs, category_counts):
    """
    Spec 7.3: provenance report for the India-centric instrument. Documents the
    base dataset, the multilingual extension, its validation status, and the
    fallback recommendation. Printed at startup and embedded verbatim in the
    paper's data section; the extension must be defensible to reviewers.
    """
    os.makedirs(prov_dir, exist_ok=True)
    report = {
        'instrument': 'Indian-context bias (English subset)',
        'base_dataset': {
            'name': 'Indian-BhED',
            'citation': 'Khandelwal, K. et al. (2023). Indian-BhED. arXiv:2309.08573.',
            'language': 'English',
            'role': 'PRIMARY',
        },
        'extension_source': {
            'name': 'Indian-Multilingual-Bias-Dataset',
            'repo': base_repo,
            'relationship': ('Multilingual extension of Indian-BhED; the English '
                             'subset is used here as a secondary confirmation '
                             'instrument alongside the original Indian-BhED English.'),
            'role': 'SUPPLEMENTARY',
        },
        'validation_status': 'SUPPLEMENTARY',
        'inter_annotator_agreement': {
            'english_subset_kappa': None,
            'note': ('English-subset inter-annotator agreement (Cohen/Fleiss kappa) '
                     'was not re-measured for the extension in this pipeline. '
                     'Treat the extension as supplementary until measured.'),
        },
        'recommendation': ('If the English-subset IAA kappa < 0.6, use the original '
                           'Indian-BhED English as the primary India-centric '
                           'instrument and demote this extension to supplementary. '
                           'The primary confirmatory instrument for the paper remains '
                           'Multi-CrowS-Pairs English; Indian-context results are '
                           'reported as secondary confirmation, never gated on.'),
        'categories': category_counts,
        'total_pairs': n_pairs,
        'per_category_files_dir': 'datasets_eval/indian_bias/english/',
    }
    prov_path = os.path.join(prov_dir, 'provenance_report.json')
    with open(prov_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    logger.info("Indian-bias provenance report (spec 7.3):\n%s",
                json.dumps(report, indent=2))
    logger.info(f"Provenance report written to {prov_path}")


def download_indian_bias(namespace: str = 'Debk'):
    """
    Download the Indian bias instrument (Indian-BhED base + multilingual
    extension) and write the spec-7.2 deliverables:
      - datasets_eval/indian_bias/english/{Caste,Gender,India_Religious,Race}.csv
      - datasets_eval/indian_bias/provenance/provenance_report.json
    The merged datasets_eval/indian_bias/indian_bias_english.csv is also kept:
    the Stage 1-3 bias evaluators read the merged file (per-category rows carry
    their Category column, so per-category preference rates are recovered
    downstream from the single eval pass).
    """
    out_dir = os.path.join(EVAL_DIR, 'indian_bias')
    english_dir = os.path.join(out_dir, 'english')
    prov_dir = os.path.join(out_dir, 'provenance')
    os.makedirs(english_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'indian_bias_english.csv')

    per_category_present = all(
        os.path.exists(os.path.join(english_dir, fn))
        for fn in _INDIAN_CATEGORY_FILES.values())
    if os.path.exists(out_path) and per_category_present and \
            os.path.exists(os.path.join(prov_dir, 'provenance_report.json')):
        logger.info(f"Indian bias deliverables already present under {out_dir}")
        return

    df = None
    used_repo = None
    for repo in [f"{namespace}/Indian-Multilingual-Bias-Dataset",
                 "Aksht/Indian-Multilingual-Bias-English"]:
        try:
            logger.info(f"Downloading Indian bias dataset from {repo}...")
            dataset = load_dataset(repo, split="train")
            df = dataset.to_pandas()
            used_repo = repo
            break
        except Exception as e:
            logger.warning(f"Could not load {repo}: {e}")

    if df is None:
        logger.error("Indian bias dataset could not be downloaded from any source. "
                     "The pipeline can proceed on Multi-CrowS-Pairs only, but the "
                     "India-centric analysis will be missing.")
        return

    df = _standardize_pair_columns(df, 'Indian bias dataset')

    # Normalise the category column into the 4 canonical categories.
    src_cat_col = 'bias_type' if 'bias_type' in df.columns else (
        'category' if 'category' in df.columns else None)
    if src_cat_col is None:
        logger.warning("Indian bias dataset has no category column; all pairs "
                       "labelled 'unknown' and per-category split will be partial.")
        df['category'] = 'unknown'
        src_cat_col = 'category'
    df['category'] = df[src_cat_col].map(_canonical_indian_category)

    # Merged file (consumed by the bias evaluators).
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} pairs to {out_path}")

    # Per-category files (spec 7.2 deliverable).
    category_counts = {}
    for category, filename in _INDIAN_CATEGORY_FILES.items():
        sub = df[df['category'] == category]
        cat_path = os.path.join(english_dir, filename)
        sub.to_csv(cat_path, index=False)
        category_counts[category] = int(len(sub))
        logger.info(f"  {category}: {len(sub)} pairs -> {cat_path}")
    uncategorised = int((~df['category'].isin(_INDIAN_CATEGORY_FILES)).sum())
    if uncategorised:
        logger.warning(f"Indian bias: {uncategorised} pairs did not map to one of "
                       f"the 4 canonical categories (kept only in the merged file).")

    # Provenance report (spec 7.3 deliverable).
    _write_indian_provenance(prov_dir, used_repo, int(len(df)), category_counts)


def _detokenize(tokens):
    """Join a WinoBias token list into a readable sentence."""
    out = ''
    no_space_before = {'.', ',', '!', '?', ';', ':', "'s", "n't", "'", ")", "]"}
    for tok in tokens:
        if out and tok not in no_space_before and not tok.startswith("'"):
            out += ' '
        out += tok
    return out


def download_winobias():
    """
    Download WinoBias and preprocess each example into the masked-pronoun
    evaluation format: columns 'sentence' (full text) and 'pronoun' (the gold
    gendered pronoun found in the sentence).
    """
    out_dir = os.path.join(EVAL_DIR, 'winobias')
    os.makedirs(out_dir, exist_ok=True)

    for subset in ['type1_pro', 'type1_anti', 'type2_pro', 'type2_anti']:
        out_path = os.path.join(out_dir, f'winobias_{subset}.csv')
        if os.path.exists(out_path):
            continue

        logger.info(f"Downloading WinoBias {subset}...")
        try:
            # wino_bias is a script-based dataset: datasets==2.16 requires
            # trust_remote_code=True or the download raises and the CSVs
            # silently never materialise.
            dataset = load_dataset("wino_bias", subset, split="test",
                                   trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to download WinoBias {subset}: {e}")
            continue

        rows = []
        skipped = 0
        for ex in dataset:
            tokens = ex.get('tokens') or []
            pronouns = [t for t in tokens if t.lower() in _GENDERED_PRONOUNS]
            if len(pronouns) != 1:
                # Ambiguous or missing pronoun; skip rather than guess
                skipped += 1
                continue
            rows.append({
                'sentence': _detokenize(tokens),
                'pronoun': pronouns[0],
            })

        if not rows:
            logger.error(f"WinoBias {subset}: no usable examples produced.")
            continue

        pd.DataFrame(rows).to_csv(out_path, index=False)
        logger.info(f"Saved {len(rows)} WinoBias {subset} examples "
                    f"({skipped} skipped for ambiguous pronouns) to {out_path}")


def create_mlm_validation_set():
    """
    Extract 10,000 documents from train_raw.jsonl to act as the fixed
    validation set. Removes them from train_raw to prevent leakage.
    """
    train_path = os.path.join(DATA_DIR, 'fineweb-edu', 'train_raw.jsonl')
    val_path = os.path.join(DATA_DIR, 'fineweb-edu', 'validation.jsonl')

    if os.path.exists(val_path):
        logger.info(f"Validation set already exists at {val_path}")
        return

    if not os.path.exists(train_path):
        logger.error(f"Training corpus not found at {train_path}. Cannot create validation set.")
        return

    logger.info("Creating fixed validation set (10,000 docs)...")

    val_docs = []
    temp_train_path = train_path + ".tmp"

    with open(train_path, 'r', encoding='utf-8') as f_in, \
         open(temp_train_path, 'w', encoding='utf-8') as f_out:

        for i, line in enumerate(f_in):
            if i < 10000:
                val_docs.append(line)
            else:
                f_out.write(line)

    with open(val_path, 'w', encoding='utf-8') as f_val:
        f_val.writelines(val_docs)

    os.replace(temp_train_path, train_path)
    logger.info(f"Saved {len(val_docs)} docs to {val_path} and removed them from train_raw.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-namespace', type=str, default='Debk',
                        help='HF org prefix for project datasets (spec 7.2)')
    args = parser.parse_args()

    logger.info("Starting evaluation datasets download...")
    download_multi_crows_pairs(args.dataset_namespace)
    download_indian_bias(args.dataset_namespace)
    download_winobias()
    create_mlm_validation_set()
    logger.info("Evaluation datasets download complete.")

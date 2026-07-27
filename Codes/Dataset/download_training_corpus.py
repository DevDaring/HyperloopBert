import os
import sys
import json
import logging
from datasets import load_dataset
from tqdm import tqdm

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.env_loader import env_loader

logger = setup_logging('download_training_corpus')

def download_fineweb_edu(output_path: str, target_docs: int = 5_000_000):
    """
    Stream FineWeb-Edu (sample-10BT subset) and save the first target_docs
    to a local JSONL file. 5M documents is roughly 2.5B tokens, which is 
    plenty for our max 500M token Stage 3 budget.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # RERUN-LEAKAGE GUARD: once the validation split has been carved out of
    # train_raw.jsonl, re-downloading would silently re-include the validation
    # documents in training (the carve step refuses to re-run because
    # validation.jsonl exists). Hard-stop instead of corrupting the split.
    val_path = os.path.join(os.path.dirname(output_path), 'validation.jsonl')
    if os.path.exists(output_path) and os.path.exists(val_path):
        logger.error(
            f"{output_path} already exists AND the validation split has been "
            f"carved out. Re-downloading would re-include validation documents "
            f"in training (leakage). If you truly intend to rebuild the corpus, "
            f"delete BOTH {output_path} and {val_path} first.")
        raise SystemExit(1)
    if os.path.exists(output_path):
        logger.info(f"{output_path} already exists; skipping download.")
        return

    hf_token = env_loader.get('HF_KEY')
    if not hf_token:
        logger.warning("HF_KEY not found in environment. Attempting download without authentication.")
        
    logger.info(f"Connecting to HuggingFaceFW/fineweb-edu (sample-10BT)...")
    
    # Use streaming to avoid downloading the entire 10BT subset to disk
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        token=hf_token
    )
    
    logger.info(f"Downloading {target_docs} documents to {output_path}")
    
    docs_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        with tqdm(total=target_docs, desc="Downloading FineWeb-Edu") as pbar:
            for item in dataset:
                # We only need the text field for pretraining
                if 'text' in item and item['text'].strip():
                    doc = {'text': item['text'], 'id': item.get('id', str(docs_written))}
                    f.write(json.dumps(doc) + '\n')
                    docs_written += 1
                    pbar.update(1)
                    
                if docs_written >= target_docs:
                    break
                    
    logger.info(f"Successfully downloaded {docs_written} documents.")
    
if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    OUTPUT_PATH = os.path.join(DATA_DIR, 'fineweb-edu', 'train_raw.jsonl')
    
    # 5M docs * ~500 tokens/doc = ~2.5B tokens
    # This leaves plenty of headroom for contamination filtering and the 500M stage 3 max budget.
    download_fineweb_edu(OUTPUT_PATH, target_docs=5_000_000)

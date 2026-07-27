import os
import sys

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.integrity import run_integrity_suite, check_eval_duplicates, build_manifest

logger = setup_logging('validate_and_manifest')

def main():
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    EVAL_DIR = os.path.join(DATA_DIR, 'datasets_eval')
    MANIFEST_PATH = os.path.join(DATA_DIR, 'dataset_manifest.json')
    QUARANTINE_DIR = os.path.join(DATA_DIR, 'quarantine')
    
    logger.info("Starting Dataset Validation and Manifest Generation...")
    
    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    
    # 1. Run full integrity suite (removes corruption, removes duplicates)
    logger.info("Running JSONL integrity checks...")
    summary = run_integrity_suite(DATA_DIR, MANIFEST_PATH, logger=logger, quarantine_dir=QUARANTINE_DIR)
    
    logger.info(f"Integrity Check Summary: "
                f"Corrupt files: {summary['corrupt_files']}, "
                f"Total quarantined lines: {summary['total_quarantined']}, "
                f"Duplicate files: {summary['duplicate_files']}, "
                f"Duplicates removed: {summary['total_duplicates_removed']}")
                
    # 2. Check evaluation pairs for duplicates
    logger.info("Checking Evaluation CSVs for duplicate pairs...")
    eval_dups = check_eval_duplicates(EVAL_DIR, logger=logger)
    if not eval_dups:
        logger.info("No duplicate evaluation pairs found. Perfect.")
        
    # 3. Build Manifest
    logger.info("Building finalized manifest...")
    
    files_to_manifest = []
    for root, _, files in os.walk(DATA_DIR):
        if 'quarantine' in root:
            continue
        for file in files:
            # Tokenizer files are manifested too: an accidental tokenizer
            # rebuild would silently shift every downstream number.
            in_tokenizer_dir = os.path.basename(root) == 'tokenizer'
            if file.endswith(('.jsonl', '.csv')) or (in_tokenizer_dir and file.endswith('.json')):
                files_to_manifest.append(os.path.join(root, file))
                
    manifest = build_manifest(files_to_manifest, MANIFEST_PATH)
    logger.info(f"Manifest built at {MANIFEST_PATH} with {len(manifest)} files tracked.")
    
    logger.info("Validation complete.")

if __name__ == "__main__":
    main()

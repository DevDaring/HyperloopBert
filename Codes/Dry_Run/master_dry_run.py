import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging

logger = setup_logging('master_dry_run')

def main():
    logger.info("=== STARTING MASTER DRY RUN ===")
    
    # Check tokenizer exists
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    tokenizer_dir = os.path.join(data_dir, 'tokenizer')
    if not os.path.exists(tokenizer_dir):
        logger.error(f"Tokenizer not found at {tokenizer_dir}. Please run Dataset/ scripts first.")
        return
        
    try:
        import Dry_Run.dry_run_stage1 as s1
        logger.info("--- STAGE 1 ---")
        s1.main()
        
        import Dry_Run.dry_run_stage2 as s2
        logger.info("--- STAGE 2 ---")
        s2.main()
        
        import Dry_Run.dry_run_stage3 as s3
        logger.info("--- STAGE 3 ---")
        s3.main()
        
        logger.info("=== MASTER DRY RUN COMPLETED SUCCESSFULLY ===")
    except Exception as e:
        logger.error(f"Master Dry Run failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

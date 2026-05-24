import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging

logger = setup_logging('dry_run_stage1')

def patch_config():
    import Stage1.config_stage1 as cfg1
    import Dry_Run.config_dry_run as cfg_dry
    
    logger.info("Monkeypatching Stage 1 config for dry run...")
    cfg1.SIZES = cfg_dry.SIZES
    cfg1.DEFAULT_SEEDS = cfg_dry.DEFAULT_SEEDS
    cfg1.MAX_TOKENS = cfg_dry.MAX_TOKENS
    cfg1.TOKEN_MARKERS = cfg_dry.TOKEN_MARKERS.copy()
    cfg1.VAL_EVERY_STEPS = cfg_dry.VAL_EVERY_STEPS
    cfg1.CHECKPOINT_EVERY_STEPS = cfg_dry.CHECKPOINT_EVERY_STEPS
    cfg1.VAL_SAMPLES = cfg_dry.VAL_SAMPLES
    cfg1.DEFAULT_ISO_BANDS = cfg_dry.DEFAULT_ISO_BANDS.copy()
    cfg1.ARCHITECTURES = cfg_dry.STAGE1_ARCHITECTURES

def main():
    patch_config()
    
    # Run training
    logger.info("Starting Stage 1 Dry Run Training...")
    import Stage1.train_stage1 as train1
    # Mock sys.argv
    sys.argv = ['train_stage1.py', '--seeds', '1', '--sizes', 'tiny']
    train1.main()
    
    # Run Eval Bias
    logger.info("Starting Stage 1 Dry Run Eval Bias...")
    import Stage1.eval_bias_stage1 as eval_bias1
    sys.argv = ['eval_bias_stage1.py']
    eval_bias1.main()
    
    # Run Analyze
    logger.info("Starting Stage 1 Dry Run Analysis...")
    import Stage1.analyze_stage1 as analyze1
    sys.argv = ['analyze_stage1.py']
    analyze1.main()
    
    logger.info("Stage 1 Dry Run Complete!")

if __name__ == "__main__":
    main()

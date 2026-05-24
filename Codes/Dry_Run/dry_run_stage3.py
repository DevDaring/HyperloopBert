import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging

logger = setup_logging('dry_run_stage3')

def patch_config():
    import Stage3.config_stage3 as cfg3
    import Dry_Run.config_dry_run as cfg_dry
    
    logger.info("Monkeypatching Stage 3 config for dry run...")
    cfg3.SIZES = cfg_dry.SIZES
    cfg3.DEFAULT_SEEDS = cfg_dry.DEFAULT_SEEDS
    cfg3.MAX_TOKENS = cfg_dry.MAX_TOKENS
    cfg3.TOKEN_MARKERS = cfg_dry.TOKEN_MARKERS.copy()
    cfg3.VAL_EVERY_STEPS = cfg_dry.VAL_EVERY_STEPS
    cfg3.VAL_SAMPLES = cfg_dry.VAL_SAMPLES
    cfg3.DEFAULT_ISO_BANDS = cfg_dry.DEFAULT_ISO_BANDS.copy()
    cfg3.ARCHITECTURES = cfg_dry.STAGE3_ARCHITECTURES
    cfg3.GLUE_EPOCHS = cfg_dry.GLUE_EPOCHS
    cfg3.GLUE_BATCH_SIZE = cfg_dry.GLUE_BATCH_SIZE
    # Less ablation for dry run
    cfg3.NUM_STREAMS_ABLATION = [2, 4] 

def main():
    patch_config()
    
    # Run training
    logger.info("Starting Stage 3 Dry Run Training...")
    import Stage3.train_stage3 as train3
    sys.argv = ['train_stage3.py', '--seeds', '1', '--sizes', 'tiny']
    train3.main()
    
    # Run Eval Bias
    logger.info("Starting Stage 3 Dry Run Eval Bias...")
    import Stage3.eval_bias_stage3 as eval_bias3
    sys.argv = ['eval_bias_stage3.py']
    eval_bias3.main()
    
    # Run Eval GLUE
    logger.info("Starting Stage 3 Dry Run Eval GLUE...")
    import Stage3.eval_glue_stage3 as eval_glue3
    sys.argv = ['eval_glue_stage3.py']
    eval_glue3.main()
    
    # Run Stream Analysis
    logger.info("Starting Stage 3 Dry Run Stream Analysis...")
    import Stage3.stream_analysis_stage3 as stream3
    sys.argv = ['stream_analysis_stage3.py']
    stream3.main()
    
    # Run Analyze
    logger.info("Starting Stage 3 Dry Run Analysis...")
    import Stage3.analyze_stage3 as analyze3
    sys.argv = ['analyze_stage3.py']
    analyze3.main()
    
    logger.info("Stage 3 Dry Run Complete!")

if __name__ == "__main__":
    main()

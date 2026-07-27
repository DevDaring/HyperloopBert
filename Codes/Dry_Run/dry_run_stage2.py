import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging

logger = setup_logging('dry_run_stage2')

def patch_config():
    import Stage2.config_stage2 as cfg2
    import Dry_Run.config_dry_run as cfg_dry
    
    logger.info("Monkeypatching Stage 2 config for dry run...")
    cfg2.SIZES = cfg_dry.SIZES
    cfg2.DEFAULT_SEEDS = cfg_dry.DEFAULT_SEEDS
    cfg2.MAX_TOKENS = cfg_dry.MAX_TOKENS
    cfg2.TOKEN_MARKERS = cfg_dry.TOKEN_MARKERS.copy()
    cfg2.VAL_EVERY_STEPS = cfg_dry.VAL_EVERY_STEPS
    cfg2.CHECKPOINT_EVERY_STEPS = cfg_dry.CHECKPOINT_EVERY_STEPS
    cfg2.VAL_SAMPLES = cfg_dry.VAL_SAMPLES
    cfg2.DEFAULT_ISO_BANDS = cfg_dry.DEFAULT_ISO_BANDS.copy()
    cfg2.ARCHITECTURES = cfg_dry.STAGE2_ARCHITECTURES
    cfg2.GLUE_EPOCHS = cfg_dry.GLUE_EPOCHS
    cfg2.GLUE_BATCH_SIZE = cfg_dry.GLUE_BATCH_SIZE
    cfg2.EXTERNAL_MODELS = cfg_dry.EXTERNAL_MODELS

    # CRITICAL: isolate from real Stage 2 output directories (see dry_run_stage1
    # for why -- 'tiny' is also a real Stage 2 size).
    sub = cfg_dry.DRY_RUN_SUBDIR
    cfg2.RESULTS_DIR = f'results/{sub}/stage2'
    cfg2.MODELS_DIR = f'models/{sub}/stage2'
    cfg2.FIGURES_DIR = f'figures/{sub}/stage2'
    cfg2.CHECKPOINTS_DIR = f'checkpoints/{sub}/stage2'

def main():
    patch_config()
    
    # Run training
    logger.info("Starting Stage 2 Dry Run Training...")
    import Stage2.train_stage2 as train2
    sys.argv = ['train_stage2.py', '--seeds', '1', '--sizes', 'tiny']
    train2.main()
    
    # Run Eval Bias
    logger.info("Starting Stage 2 Dry Run Eval Bias...")
    import Stage2.eval_bias_stage2 as eval_bias2
    sys.argv = ['eval_bias_stage2.py']
    eval_bias2.main()
    
    # Run Eval GLUE
    logger.info("Starting Stage 2 Dry Run Eval GLUE...")
    import Stage2.eval_glue_stage2 as eval_glue2
    sys.argv = ['eval_glue_stage2.py']
    eval_glue2.main()
    
    # Run External Calibration
    logger.info("Starting Stage 2 Dry Run External Calibration...")
    import Stage2.external_calibration_stage2 as external2
    sys.argv = ['external_calibration_stage2.py']
    external2.main()
    
    # Run Loop Trajectory
    logger.info("Starting Stage 2 Dry Run Loop Trajectory...")
    import Stage2.loop_trajectory_stage2 as loop2
    sys.argv = ['loop_trajectory_stage2.py']
    loop2.main()
    
    # Run Analyze
    logger.info("Starting Stage 2 Dry Run Analysis...")
    import Stage2.analyze_stage2 as analyze2
    sys.argv = ['analyze_stage2.py']
    analyze2.main()
    
    logger.info("Stage 2 Dry Run Complete!")

if __name__ == "__main__":
    main()

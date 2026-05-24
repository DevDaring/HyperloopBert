import os

# Stage 3 Configuration
STAGE = 'stage3'
ARCHITECTURES = ['HyperloopBERT', 'EarlyMergeHyperloopBERT']
SIZES = ['base']  # Only base scale for Stage 3
DEFAULT_SEEDS = [42, 43, 44]  # 3 seeds for robust measurement

MAX_TOKENS = 400_000_000  # 400M tokens
TOKEN_MARKERS = [100_000_000, 200_000_000, 400_000_000]
SEQ_LENGTH = 128

# Training Hyperparameters
LEARNING_RATE = 5e-4
ADAMW_BETAS = (0.9, 0.98)
ADAMW_EPS = 1e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
EFFECTIVE_BATCH_SIZE = 64
MICRO_BATCH_SIZE = 16

# Checkpointing & Validation
VAL_EVERY_STEPS = 2000
VAL_SAMPLES = 5000

# MLM Params
MLM_PROBABILITY = 0.15
MLM_MASK_PROB = 0.80
MLM_RANDOM_PROB = 0.10
MLM_KEEP_PROB = 0.10

# Quality Screen
PSEUDO_PERPLEXITY_QUALITY_THRESHOLD = 60.0

# Iso-loss Bands
DEFAULT_ISO_BANDS = [3.7, 3.4, 3.1] # Adjusted for Stage 3

# Eval Params
BIAS_EVAL_BATCH_SIZE = 50
VAL_DATASET_NAMESPACE = 'Debk'

# GLUE fine-tuning params
GLUE_TASKS = ['sst2', 'rte']
GLUE_LR = 2e-5
GLUE_EPOCHS = 3
GLUE_BATCH_SIZE = 32

# Hyperloop specific
DEFAULT_NUM_STREAMS = 4
NUM_STREAMS_ABLATION = [1, 2, 4, 8]  # Ablation over stream counts

# Directories (relative to project root)
RESULTS_DIR = 'results/stage3'
MODELS_DIR = 'models/stage3'
FIGURES_DIR = 'figures/stage3'
DEFAULT_NUM_STREAMS = 4

# Directories (relative to project root)
RESULTS_DIR = 'results/stage3'
FIGURES_DIR = 'figures/stage3'
CHECKPOINTS_DIR = 'checkpoints/stage3'
MODELS_DIR = 'models/stage3'

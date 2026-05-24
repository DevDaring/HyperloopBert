import os

# Stage 1 Configuration
STAGE = 'stage1'
ARCHITECTURES = ['VanillaBERT', 'LoopedBERT']
SIZES = ['tiny', 'small', 'base']  # 'base' maps to 'base-ish' in the spec
DEFAULT_SEEDS = [42]  # --seeds 1 default; --seeds 3 gives [42,43,44]

MAX_TOKENS = 200_000_000  # 200M token budget for Stage 1
TOKEN_MARKERS = [50_000_000, 100_000_000, 200_000_000]  # for secondary endpoint
SEQ_LENGTH = 128

# Training Hyperparameters
LEARNING_RATE = 5e-4
ADAMW_BETAS = (0.9, 0.98)
ADAMW_EPS = 1e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
EFFECTIVE_BATCH_SIZE = 64
MICRO_BATCH_SIZE = 16  # will be halved on OOM, floor=4

# Checkpointing & Validation
VAL_EVERY_STEPS = 2000
CHECKPOINT_EVERY_STEPS = 2000
VAL_SAMPLES = 5000

# MLM Params
MLM_PROBABILITY = 0.15
MLM_MASK_PROB = 0.80
MLM_RANDOM_PROB = 0.10
MLM_KEEP_PROB = 0.10

# Quality Screen
PSEUDO_PERPLEXITY_QUALITY_THRESHOLD = 60.0

# Iso-loss Bands
DEFAULT_ISO_BANDS = [4.0, 3.7, 3.4, 3.1]

# Eval Params
BIAS_EVAL_BATCH_SIZE = 50  # rows written per CSV append
VAL_DATASET_NAMESPACE = 'Debk'

# Directories (relative to project root)
RESULTS_DIR = 'results/stage1'
MODELS_DIR = 'models/stage1'
FIGURES_DIR = 'figures/stage1'
CHECKPOINTS_DIR = 'checkpoints/stage1'
MODELS_DIR = 'models/stage1'

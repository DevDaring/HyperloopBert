import os

# Stage 2 Configuration
STAGE = 'stage2'
# VanillaBERT6 = 6-layer Vanilla, parameter-matched to LoopedBERT's unique
# parameter count (NOT compute-matched). Exploratory control for the
# "looped is just a smaller model" objection; excluded from the confirmatory
# family and from primary-band computation.
ARCHITECTURES = ['VanillaBERT', 'LoopedBERT', 'ALBERTLoopedBERT']
EXPLORATORY_ARCHITECTURES = ['VanillaBERT6']
SIZES = ['small', 'base']  # base-ish primary + small scale check
DEFAULT_SEEDS = [42, 43, 44, 45, 46]  # supports --seeds up to 5

MAX_TOKENS = 400_000_000  # 400M token budget for Stage 2
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
BIAS_EVAL_BATCH_SIZE = 50
VAL_DATASET_NAMESPACE = 'Debk'

# GLUE fine-tuning params
GLUE_TASKS = ['sst2', 'rte']  # SST-2 + RTE sufficient for Stage 2
GLUE_LR = 2e-5
GLUE_EPOCHS = 3
GLUE_BATCH_SIZE = 32

# External calibration models.
# roberta-base replaces answerdotai/ModernBERT-base: ModernBERT requires
# transformers >= 4.48 (project pin: 4.46.0) and would silently vanish from
# the calibration evidence; roberta-base loads under the pin and has a
# published CrowS-Pairs score (Nangia et al. 2020) to calibrate against.
EXTERNAL_MODELS = ['bert-base-uncased', 'albert-base-v2', 'roberta-base']

# Iso-loss matching tolerance for contrasts: pairs whose actual validation
# losses differ by more than this (in nats) are flagged in the analysis
# output. Reported, not silently dropped.
ISO_LOSS_TOLERANCE = 0.05

# Adaptive validation cadence near an uncrossed band (bounds band overshoot)
VAL_FINE_EVERY_STEPS = 500
VAL_FINE_MARGIN = 0.15

# GLUE quality screen (spec 9.1): configs with GLUE average below this are
# flagged and excluded from confirmatory contrasts.
GLUE_QUALITY_SCREEN = 55.0

# Directories (relative to project root)
RESULTS_DIR = 'results/stage2'
MODELS_DIR = 'models/stage2'
FIGURES_DIR = 'figures/stage2'
CHECKPOINTS_DIR = 'checkpoints/stage2'

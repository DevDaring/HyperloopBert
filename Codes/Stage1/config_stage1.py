import os

# Stage 1 Configuration
STAGE = 'stage1'
ARCHITECTURES = ['VanillaBERT', 'LoopedBERT']
SIZES = ['tiny', 'small', 'base']  # 'base' maps to 'base-ish' in the spec

# Ordered seed pool: train_stage1 --seeds N runs the first N seeds.
# The Stage 1 decision rule caps EXTEND-SEEDS at 3 seeds; entries beyond the
# third exist only for consistency with Stage 2's pool (config_stage2.py).
SEED_POOL = [42, 43, 44, 45, 46]
DEFAULT_SEEDS = SEED_POOL[:1]

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

# Capability gate (undertrained-model guard) -- THREE pre-registered legs,
# all evaluated in analyze_stage1.py before ANY contrast is interpreted:
#   Leg 1: GLUE screen pulled forward from Stage 2 -- SST-2 + RTE fine-tuning
#          accuracy above chance (one-sided exact binomial, alpha below).
#   Leg 2: baseline bias detectable on VanillaBERT base (item-level preference
#          rate bootstrap CI excludes 0.5 from above).
#   Leg 3: gendered-coreference signal control -- WinoBias masked-pronoun
#          accuracy on the pro-stereotypical splits above chance (binomial).
#          On pro items, stereotype knowledge and coreference ability point the
#          same way, so this is the most sensitive detector that ANY gendered
#          pronoun signal was learned; at-chance means WinoBias in Stage 2
#          would measure noise.
# Legs 1 & 3 are produced by Stage1/eval_capability_stage1.py.
GLUE_TASKS = ['sst2', 'rte']
GLUE_LR = 2e-5
GLUE_EPOCHS = 3
GLUE_BATCH_SIZE = 32
CAPABILITY_ALPHA = 0.05

# Iso-loss Bands
DEFAULT_ISO_BANDS = [4.0, 3.7, 3.4, 3.1]

# Iso-loss matching tolerance for contrasts (nats); gaps above this are flagged
ISO_LOSS_TOLERANCE = 0.05

# Adaptive validation cadence near an uncrossed band (bounds band overshoot)
VAL_FINE_EVERY_STEPS = 500
VAL_FINE_MARGIN = 0.15

# Eval Params
BIAS_EVAL_BATCH_SIZE = 50  # rows written per CSV append
VAL_DATASET_NAMESPACE = 'ANONYMOUS'

# Directories (relative to project root)
RESULTS_DIR = 'results/stage1'
MODELS_DIR = 'models/stage1'
FIGURES_DIR = 'figures/stage1'
CHECKPOINTS_DIR = 'checkpoints/stage1'
MODELS_DIR = 'models/stage1'

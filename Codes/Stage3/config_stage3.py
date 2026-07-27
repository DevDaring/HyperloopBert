import os

# Stage 3 Configuration
STAGE = 'stage3'

# Primary confirmatory set (pre-registered contrasts need all four):
#   Vanilla vs Hyperloop (primary), Vanilla vs Looped, Looped vs Hyperloop.
ARCHITECTURES = ['VanillaBERT', 'LoopedBERT', 'ALBERTLoopedBERT', 'HyperloopBERT']

# Ablations:
#   - Stream-count dose-response, n in {1, 2, 4} (n=4 is the primary Hyperloop).
#     All arms train at MAX_TOKENS so ONLY stream count varies (never budget).
#     n=1 is the (approximate) collapse-to-Looped sanity check; the n=4 vs n=1
#     permutation test is a confirmatory contrast.
#   - EarlyMerge (merge_at in {1,2,3}) is an EVAL-TIME OOD intervention on the
#     trained 4-stream model (stream_analysis_stage3.py) -- no training runs.
NUM_STREAMS_ABLATION = [1, 2, 4]
EARLY_MERGE_POINTS = [1, 2, 3]
ABLATION_SEEDS = [42, 43, 44]

SIZES = ['base']  # Only base scale for Stage 3
DEFAULT_SEEDS = [42, 43, 44]  # 3 seeds for robust measurement

MAX_TOKENS = 400_000_000  # 400M tokens @ seq=128 (primary phase 1)
TOKEN_MARKERS = [100_000_000, 200_000_000, 400_000_000]
SEQ_LENGTH = 128

# Sequence-length adaptation tail (spec 2.2): primary runs continue for
# TAIL_TOKENS @ SEQ_LENGTH_TAIL from the phase-1 weights, so the full primary
# budget is 400M @ seq=128 + 100M @ seq=256 = 500M tokens. Ablation arms
# (stream-count dose-response) are seq=128 only and skip the tail. Requires
# MAX_POSITION_EMBEDDINGS >= SEQ_LENGTH_TAIL (architectures.py: 512).
SEQ_LENGTH_TAIL = 256
TAIL_TOKENS = 100_000_000
TAIL_TOKEN_MARKERS = [100_000_000]
# Bands for the seq=256 tail (typically shallower than phase 1 after the extra
# adaptation tokens; kept equal here and matched per-arm by the iso-band tracker).
TAIL_ISO_BANDS = [3.7, 3.4, 3.1]

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
DEFAULT_ISO_BANDS = [3.7, 3.4, 3.1]  # Adjusted for Stage 3

# Iso-loss matching tolerance for contrasts (nats); gaps above this are flagged
ISO_LOSS_TOLERANCE = 0.05

# Adaptive validation cadence near an uncrossed band (bounds band overshoot)
VAL_FINE_EVERY_STEPS = 500
VAL_FINE_MARGIN = 0.15

# Eval Params
BIAS_EVAL_BATCH_SIZE = 50
VAL_DATASET_NAMESPACE = 'Debk'

# GLUE fine-tuning params
GLUE_TASKS = ['sst2', 'mrpc', 'qnli', 'rte']  # full screen for the final paper
GLUE_LR = 2e-5
GLUE_EPOCHS = 3
GLUE_BATCH_SIZE = 32

# Hyperloop specific
DEFAULT_NUM_STREAMS = 4

# Directories (relative to project root)
RESULTS_DIR = 'results/stage3'
MODELS_DIR = 'models/stage3'
FIGURES_DIR = 'figures/stage3'
CHECKPOINTS_DIR = 'checkpoints/stage3'

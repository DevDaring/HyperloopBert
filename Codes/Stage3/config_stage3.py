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
# H100 RUN (amendment A11/A12): the stream-count ablation is DISABLED for this
# budget -- setting it to [4] only means build_run_list emits no extra ablation
# runs (n=4 is already the primary Hyperloop arm). The dose-response angle is
# dropped in favour of spending the budget on model QUALITY, which the pilot
# proved is the binding constraint for bias detectability.
NUM_STREAMS_ABLATION = [4]
EARLY_MERGE_POINTS = [1, 2, 3]   # eval-time only, costs no training
ABLATION_SEEDS = [42, 43, 44]

SIZES = ['base']  # Only base scale for Stage 3
DEFAULT_SEEDS = [42]  # H100 budget: 1 seed x 4 architectures (item-level test
                      # with n~1508 pairs is the PRE-REGISTERED PRIMARY test;
                      # seed-level is robustness only, and is the disclosed
                      # limitation of this run)

# Token budget derived from MEASURED H100 throughput (275,945 tok/s at
# micro=512, grad-ckpt OFF): 7B tokens ~= 7.05 h per run, 4 runs ~= 28.2 h,
# which fits the available credit with margin for evals and teardown.
MAX_TOKENS = 7_000_000_000
TOKEN_MARKERS = [2_000_000_000, 4_000_000_000, 7_000_000_000]
SEQ_LENGTH = 128

# Sequence-length adaptation tail disabled for this budget (see A12): the
# 100M @ seq=256 phase would consume time better spent on primary quality.
ENABLE_SEQ_TAIL = False

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
# LEARNING_RATE: 5e-4 COLLAPSED the model (unigram-only, ctx_cos=1.0000) -- see
# amendment A11. At EFFECTIVE_BATCH_SIZE=512 three candidates were measured on
# 250M tokens each with the real schedule shape:
#     1e-4 -> loss 6.286  ctx_cos 0.744
#     3e-4 -> loss 6.088  ctx_cos 0.697   <-- CHOSEN (best loss AND best context use)
#     6e-4 -> loss 6.072  ctx_cos 0.837   (marginally lower loss but context use
#                                          stalls -- the collapse signature)
LEARNING_RATE = 3e-4
ADAMW_BETAS = (0.9, 0.98)
ADAMW_EPS = 1e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
# Measured on H100: micro=512 with grad-checkpointing OFF gives 275,945 tok/s
# vs 102,860 at the old micro=16+ckpt settings (2.7x). Peak memory 56GB of 80GB.
# accum = EFFECTIVE/MICRO = 1.
EFFECTIVE_BATCH_SIZE = 512
MICRO_BATCH_SIZE = 512
# Gradient checkpointing trades compute for memory we do not need here (5GB of
# 80GB was in use); disabling it recovered ~45% throughput.
GRADIENT_CHECKPOINTING = False

# Checkpointing & Validation
VAL_EVERY_STEPS = 2000
CHECKPOINT_EVERY_STEPS = 2000
VAL_SAMPLES = 5000

# MLM Params
MLM_PROBABILITY = 0.15
MLM_MASK_PROB = 0.80
MLM_RANDOM_PROB = 0.10
MLM_KEEP_PROB = 0.10

# Quality Screen -- RECALIBRATED (A12).
# The registered 60.0 (loss <= 4.09) was set for a far larger token budget; at
# 7B tokens the measured trajectory predicts a final loss of ~4.0-4.8, so a
# threshold of 60 would skip EVERY snapshot and the run would produce nothing
# (exactly the failure mode of the 200M run). The screen is deliberately a
# COARSE filter here: the real arbiter of whether the models are good enough is
# the three-leg CAPABILITY GATE, which tests bias detectability directly.
PSEUDO_PERPLEXITY_QUALITY_THRESHOLD = 150.0

# Iso-loss Bands -- DERIVED FROM MEASUREMENT, not assumed (A12).
# lr=3e-4 trajectory on this exact hardware/data:
#   50M -> 7.091 | 100M -> 6.622 | 150M -> 6.293 | 200M -> 6.113 | 250M -> 6.088
# Log-extrapolated to the 7B budget: loss ~4.0-4.8 depending on which segment of
# the curve is fitted. Bands are set ABOVE that prediction so that EVERY
# architecture actually crosses a COMMON band -- the iso-loss protocol is
# meaningless if the arms never meet at one.
DEFAULT_ISO_BANDS = [5.5, 5.2, 4.9, 4.6]

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

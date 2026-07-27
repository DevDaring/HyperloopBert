# Dry Run Configuration Overrides

# Only tiny models
SIZES = ['tiny']
DEFAULT_SEEDS = [42]

# Severely limit tokens and steps
MAX_TOKENS = 50_000
TOKEN_MARKERS = [25_000, 50_000]
SEQ_LENGTH = 128

VAL_EVERY_STEPS = 5
CHECKPOINT_EVERY_STEPS = 5
VAL_SAMPLES = 50

# Fast GLUE
GLUE_EPOCHS = 1
GLUE_BATCH_SIZE = 8

# Stage 3 sequence-length adaptation tail (config_stage3.SEQ_LENGTH_TAIL/
# TAIL_TOKENS/TAIL_TOKEN_MARKERS/TAIL_ISO_BANDS): WITHOUT this override the
# dry run would run a REAL 100M-token phase-2 tail (the real Stage 3 budget),
# turning a fast smoke test into a multi-hour training run.
SEQ_LENGTH_TAIL = 128  # keep it small/fast; correctness of the seq-256 path
                       # itself is exercised by real Stage 3, not the dry run
TAIL_TOKENS = 25_000
TAIL_TOKEN_MARKERS = [25_000]
TAIL_ISO_BANDS = [10.0, 9.0]

# --- Output isolation (CRITICAL) ---------------------------------------------
# The dry run MUST NOT write into the same results/models/checkpoints/figures
# directories real Stage 1/2/3 runs use: Stage 1 and 2's 'tiny' SIZE is also
# used by the DRY RUN, so without isolation, dry-run rows (artificial iso-bands,
# 50k-token "trained" models) would land in the exact same summary_table.csv /
# iso_band_models/ paths the real run appends to and reads from -- silently
# corrupting the real experiment. Every dry_run_stageN.py patches these onto
# its stage config so ALL directory-producing code (train/eval/analyze) is
# automatically isolated with no per-script changes required.
DRY_RUN_SUBDIR = 'dry_run'


# Force early crossing of bands by setting them artificially high, 
# or just let it cross instantly if it starts high.
# For dry run, we just want it to trigger.
DEFAULT_ISO_BANDS = [10.0, 9.0] 

# Stage specific overrides
STAGE1_ARCHITECTURES = ['VanillaBERT', 'LoopedBERT']
STAGE2_ARCHITECTURES = ['VanillaBERT', 'LoopedBERT', 'ALBERTLoopedBERT']
STAGE3_ARCHITECTURES = ['HyperloopBERT', 'EarlyMergeHyperloopBERT']

# Use a fast external model for calibration
EXTERNAL_MODELS = ['prajjwal1/bert-tiny']

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

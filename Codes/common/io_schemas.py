"""
io_schemas.py
-------------
Single source of truth for all CSV column definitions in the HyperloopBert
pipeline.

Rules:
- All column names are full, unabbreviated strings.
- No abbreviations, no truncations.
- Every stage that writes a CSV must import the relevant constant from here
  and pass it to pd.DataFrame(..., columns=<CONSTANT>) or use make_empty_df().
- NEVER define column names inline in stage scripts.

Helper functions:
- validate_df_columns(df, expected_columns, allow_extra=True)
- make_empty_df(columns)
"""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Identity columns (reused across all output files)
# ---------------------------------------------------------------------------

IDENTITY_COLUMNS: List[str] = [
    "Stage",
    "Architecture",
    "Model_Size",
    "Hidden_Size",
    "Seed",
    "Unique_Parameters",
    "Total_Parameters",
    "Effective_Depth",
    "Shared_Ratio",
]

# ---------------------------------------------------------------------------
# MLM quality: results/{stage}/mlm/summary_table.csv
# ---------------------------------------------------------------------------

MLM_SUMMARY_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Stream_Count",
    "Merge_At",
    "Validation_Loss",
    "Pseudo_Perplexity",
    "Mask_Accuracy",
    "Tokens_Processed",
    "Tokens_Per_Second",
    "GPU_Hours",
    "Token_Marker",
    "Band",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Bias per-example output
# ---------------------------------------------------------------------------

BIAS_EXAMPLE_COLUMNS: List[str] = [
    "Row_Index",
    "Dataset",
    "Category",
    "Sentence_Stereotypical",
    "Sentence_AntiStereotypical",
    "PLL_Stereotypical",
    "PLL_AntiStereotypical",
    "SS_PLL_Stereotypical",
    "SS_PLL_AntiStereotypical",
    "Effect_Size",
    "Stereotype_Preferred",
] + IDENTITY_COLUMNS + [
    "Validation_Loss",
    "Band",
    "Token_Marker",
    "External_Calibration",
    "Needs_Review",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Bias summary (categories are dynamic; define base columns)
# ---------------------------------------------------------------------------

BIAS_SUMMARY_BASE_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Band",
    "Token_Marker",
    "Validation_Loss",
    "Stream_Count",
    "Merge_At",
    "Overall_Stereotype_Preference_Rate",
    "Macro_Average_Preference_Rate",
    "Mean_Effect_Size",
    "Bootstrap_CI_Low",
    "Bootstrap_CI_High",
    "PLL_SS_PLL_Agreement",
    "Scored_Row_Count",
    "Failed_Row_Count",
    "Tied_Pair_Count",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# WinoBias summary
# ---------------------------------------------------------------------------

WINOBIAS_SUMMARY_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Band",
    "Token_Marker",
    "Stream_Count",
    "Pro_Stereotype_Accuracy",
    "Anti_Stereotype_Accuracy",
    "Pro_Anti_Gap",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# External calibration
# ---------------------------------------------------------------------------

EXTERNAL_CALIBRATION_COLUMNS: List[str] = [
    "Model_Name",
    "Dataset",
    "Category",
    "Overall_Stereotype_Preference_Rate",
    # Canonical CrowS-Pairs metric (Nangia et al. 2020): preference computed on
    # the SHARED (unmodified) tokens only. Comparable to published numbers;
    # the full-sentence PLL rate above is the internal primary metric.
    "Shared_Token_Preference_Rate",
    "Macro_Average_Preference_Rate",
    "Mean_Effect_Size",
    "Scored_Row_Count",
    "Failed_Row_Count",
    "External_Calibration",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# GLUE benchmark summary
# ---------------------------------------------------------------------------

GLUE_SUMMARY_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Band",
    "Token_Marker",
    "Stream_Count",
    "Task",
    "Accuracy",
    "F1",
    # Eval-split size: the capability gate's exact binomial test needs counts
    "Eval_Example_Count",
    "GLUE_Average",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Stage 1 capability gate: WinoBias masked-pronoun control (leg 3)
# ---------------------------------------------------------------------------

WINOBIAS_CAPABILITY_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Band",
    "Split",
    "Correct_Count",
    "Scored_Count",
    "Accuracy",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Iso-loss checkpoint index
# ---------------------------------------------------------------------------

ISO_CHECKPOINT_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Band",
    "Snapshot_Path",
    "Validation_Loss_At_Snapshot",
    "Crossed_At_Step",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Mechanistic analysis: bias trajectory
# ---------------------------------------------------------------------------

BIAS_TRAJECTORY_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Dataset",
    "Category",
    "Loop_Depth",
    "Mean_Preference_Rate",
    "Std_Preference_Rate",
    "Mean_Effect_Size",
    "Trajectory_Shape",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Mechanistic analysis: representation similarity (CKA)
# ---------------------------------------------------------------------------

REP_SIMILARITY_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Loop_Pair",
    "CKA",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Mechanistic analysis: stream disagreement
# ---------------------------------------------------------------------------

STREAM_DISAGREEMENT_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Loop_Depth",
    "Stream_Disagreement",
    "Effect_Size",
    "Pearson_R",
    "Pearson_P",
    "Spearman_R",
    "Spearman_P",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Mechanistic analysis: early merge experiment
# ---------------------------------------------------------------------------

EARLY_MERGE_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Merge_At",
    "Overall_Stereotype_Preference_Rate",
    "Mean_Effect_Size",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Mechanistic analysis: demographic token drift
# ---------------------------------------------------------------------------

TOKEN_DRIFT_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Category",
    "Demographic_Term",
    "Context_Type",
    "Loop_Depth",
    "Cosine_Drift",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Mechanistic analysis: hyper-connection matrix statistics
# (stability check per Xie et al. 2025, MHC, arXiv:2512.24880)
# ---------------------------------------------------------------------------

HYPERCONNECTION_STATS_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Loop_Index",
    "Projection",
    "Stream_Index",
    "Block_Frobenius_Norm",
    "Block_Deviation_From_Init",
    "Matrix_Spectral_Norm",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Stream count ablation
# ---------------------------------------------------------------------------

STREAM_ABLATION_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Stream_Count",
    "Band",
    "Overall_Stereotype_Preference_Rate",
    "Mean_Effect_Size",
    "Std_Across_Seeds",
    "Bootstrap_CI_Low",
    "Bootstrap_CI_High",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Qualitative model-output dump (common/qualitative_output.py): the actual
# MLM-head predictions a paper reviewer expects to SEE, not just the PLL score.
# Two artifacts per snapshot:
#   (1) open-vocabulary top-k predictions at a masked position
#   (2) targeted-contrast probabilities on paired demographic tokens
# ---------------------------------------------------------------------------

QUALITATIVE_TOPK_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Band",
    "Token_Marker",
    "Stream_Count",
    "Merge_At",
    "Probe_ID",
    "Category",
    "Masked_Sentence",
    "Mask_Position",
    "Rank",
    "Predicted_Token",
    "Predicted_Probability",
    "Timestamp",
]

QUALITATIVE_CONTRAST_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Band",
    "Token_Marker",
    "Stream_Count",
    "Merge_At",
    "Probe_ID",
    "Category",
    "Masked_Sentence",
    "Target_A",
    "Probability_A",
    "Target_B",
    "Probability_B",
    "Log_Odds_A_Over_B",
    "Preferred_Target",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Corpus stereotype statistics (Dataset/corpus_stereotype_stats.py):
# benchmark-pair lexeme co-occurrence in the training corpus. Interprets null
# bias results: near-zero co-occurrence means "never learned", not
# "architecturally mitigated".
# ---------------------------------------------------------------------------

CORPUS_PAIR_STATS_COLUMNS: List[str] = [
    "Dataset",
    "Row_Index",
    "Category",
    "Stereo_Terms",
    "Anti_Terms",
    "Context_Terms",
    "Docs_With_Stereo_Term",
    "Docs_With_Anti_Term",
    "Docs_With_Context_Term",
    "Stereo_Cooccurrence_Docs",
    "Anti_Cooccurrence_Docs",
    "Sampled_Docs",
    "Timestamp",
]

CORPUS_CATEGORY_STATS_COLUMNS: List[str] = [
    "Dataset",
    "Category",
    "Pair_Count",
    "Mean_Stereo_Cooccurrence_Docs",
    "Median_Stereo_Cooccurrence_Docs",
    "Mean_Anti_Cooccurrence_Docs",
    "Zero_Stereo_Cooccurrence_Fraction",
    "Sampled_Docs",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Statistical analysis: confirmatory family
# ---------------------------------------------------------------------------

CONFIRMATORY_STATS_COLUMNS: List[str] = [
    "Contrast",
    "Metric",
    "Dataset",
    "Band",
    "Model_Size",
    "Raw_P_Value",
    "Holm_Corrected_P_Value",
    "Cohens_D",
    "Significant_At_0.05",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Statistical analysis: exploratory
# ---------------------------------------------------------------------------

EXPLORATORY_STATS_COLUMNS: List[str] = [
    "Contrast",
    "Metric",
    "Dataset",
    "Band",
    "Model_Size",
    "Mean_Delta",
    "Bootstrap_CI_Low",
    "Bootstrap_CI_High",
    "Cohens_D",
    "Note",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# Stage 1 primary contrast
# ---------------------------------------------------------------------------

PRIMARY_CONTRAST_COLUMNS: List[str] = [
    "Contrast",
    "Metric",
    "Dataset",
    "Band",
    "Model_Size",
    "Mean_Delta_Preference",
    "Bootstrap_CI_Low",
    "Bootstrap_CI_High",
    "Permutation_P_Value",
    "Cohens_D",
    "Timestamp",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def validate_df_columns(
    df: pd.DataFrame,
    expected_columns: List[str],
    allow_extra: bool = True,
) -> bool:
    """
    Check whether a DataFrame has all expected columns, logging warnings for
    any that are missing.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    expected_columns : list of str
        Column names that must be present.
    allow_extra : bool
        If False, also warn about columns present in df but not in
        expected_columns.

    Returns
    -------
    bool
        True if all expected columns are present (and, when allow_extra=False,
        no extra columns exist). False otherwise.
    """
    actual_columns = set(df.columns)
    expected_set = set(expected_columns)

    missing = expected_set - actual_columns
    extra = actual_columns - expected_set

    valid = True

    if missing:
        logger.warning(
            "DataFrame is missing %d expected column(s): %s",
            len(missing),
            sorted(missing),
        )
        valid = False

    if not allow_extra and extra:
        logger.warning(
            "DataFrame has %d unexpected extra column(s): %s",
            len(extra),
            sorted(extra),
        )
        valid = False

    return valid


def make_empty_df(columns: List[str]) -> pd.DataFrame:
    """
    Create an empty DataFrame with the specified columns and no rows.

    Parameters
    ----------
    columns : list of str
        Column names for the DataFrame.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with the given columns.
    """
    return pd.DataFrame(columns=columns)

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
    "Overall_Stereotype_Preference_Rate",
    "Macro_Average_Preference_Rate",
    "Mean_Effect_Size",
    "Bootstrap_CI_Low",
    "Bootstrap_CI_High",
    "PLL_SS_PLL_Agreement",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# WinoBias summary
# ---------------------------------------------------------------------------

WINOBIAS_SUMMARY_COLUMNS: List[str] = IDENTITY_COLUMNS + [
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
    "Macro_Average_Preference_Rate",
    "Mean_Effect_Size",
    "External_Calibration",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# GLUE benchmark summary
# ---------------------------------------------------------------------------

GLUE_SUMMARY_COLUMNS: List[str] = IDENTITY_COLUMNS + [
    "Task",
    "Accuracy",
    "F1",
    "GLUE_Average",
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

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict

# CITATION: Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
#           [bootstrap_ci implementation]
# CITATION: Good, P. (2000). Permutation Tests. [paired_permutation_test]
# NOTE: Holm-Bonferroni correction applied ONLY to the small confirmatory family.
#       Everything else is explicitly labeled exploratory with uncorrected CIs.

def bootstrap_ci(values: List[float], n_bootstrap: int = 2000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean of values.
    Returns: (mean, ci_low, ci_high)
    """
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values)
    rng = np.random.RandomState(seed)
    
    # Resample with replacement
    indices = rng.randint(0, len(arr), size=(n_bootstrap, len(arr)))
    means = np.mean(arr[indices], axis=1)
    
    mean_val = np.mean(arr)
    alpha = (1.0 - ci) / 2.0
    ci_low = np.percentile(means, alpha * 100)
    ci_high = np.percentile(means, (1.0 - alpha) * 100)
    
    return float(mean_val), float(ci_low), float(ci_high)

def paired_permutation_test(a: List[float], b: List[float], n_permutations: int = 10000, alternative: str = 'greater', seed: int = 42) -> Tuple[float, float]:
    """
    Paired permutation test: H0: mean(a-b) = 0, H1: mean(a-b) > 0 (or 'less', 'two-sided').
    a, b: paired arrays of same length
    Returns: (observed_delta, p_value)
    """
    if not a or not b or len(a) != len(b):
        return 0.0, 1.0
        
    diffs = np.array(a) - np.array(b)
    obs_mean = np.mean(diffs)
    
    rng = np.random.RandomState(seed)
    n = len(diffs)
    
    # Randomly flip signs
    signs = rng.choice([-1, 1], size=(n_permutations, n))
    perm_means = np.mean(diffs * signs, axis=1)
    
    if alternative == 'greater':
        p_val = np.mean(perm_means >= obs_mean)
    elif alternative == 'less':
        p_val = np.mean(perm_means <= obs_mean)
    else: # two-sided
        p_val = np.mean(np.abs(perm_means) >= np.abs(obs_mean))
        
    return float(obs_mean), float(p_val)

def holm_bonferroni_correction(p_values: List[Tuple[str, float]], alpha: float = 0.05) -> List[Tuple[str, float, float, bool]]:
    """
    Holm-Bonferroni step-down correction.
    p_values: list of (name, p_value) tuples
    Returns: list of (name, raw_p, corrected_p, significant) tuples
    NOTE: Apply ONLY to the small confirmatory family of contrasts.
    """
    if not p_values:
        return []
        
    # Sort by p-value ascending
    sorted_p = sorted(p_values, key=lambda x: x[1])
    m = len(sorted_p)
    
    results = []
    # keep track of previous corrected p to ensure monotonicity
    prev_corrected = 0.0 
    
    for k, (name, p) in enumerate(sorted_p):
        multiplier = m - k
        corrected_p = p * multiplier
        
        # Enforce monotonicity (step-down)
        corrected_p = max(prev_corrected, min(1.0, corrected_p))
        prev_corrected = corrected_p
        
        is_significant = corrected_p <= alpha
        results.append((name, p, corrected_p, is_significant))
        
    # Restore original order
    original_order = {name: i for i, (name, _) in enumerate(p_values)}
    results.sort(key=lambda x: original_order[x[0]])
    
    return results

def cohens_d(a: List[float], b: List[float]) -> float:
    """
    Cohen's d effect size for paired samples.
    d = mean(a-b) / std(a-b)
    Returns: float
    """
    if not a or not b or len(a) != len(b):
        return 0.0
        
    diffs = np.array(a) - np.array(b)
    std_diff = np.std(diffs, ddof=1)
    
    if std_diff == 0:
        return 0.0
        
    return float(np.mean(diffs) / std_diff)

def seed_variance_report(values_by_seed: Dict[int, float]) -> Dict[str, float]:
    """
    values_by_seed: dict {seed: float}
    Returns dict: mean, std, min, max, coefficient_of_variation
    """
    if not values_by_seed:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'coefficient_of_variation': 0.0}
        
    vals = list(values_by_seed.values())
    mean_val = np.mean(vals)
    std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    cv = (std_val / mean_val) if mean_val != 0 else 0.0
    
    return {
        'mean': float(mean_val),
        'std': float(std_val),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'coefficient_of_variation': float(cv)
    }

def pearson_and_spearman(x: List[float], y: List[float], label: str = 'exploratory/uncorrected') -> Dict:
    """
    Compute both Pearson and Spearman correlations.
    Returns dict: pearson_r, pearson_p, spearman_r, spearman_p, label
    Both are labeled as exploratory/uncorrected in the output.
    """
    if not x or not y or len(x) != len(y) or len(x) < 2:
        return {
            'pearson_r': 0.0, 'pearson_p': 1.0,
            'spearman_r': 0.0, 'spearman_p': 1.0,
            'label': label
        }
        
    p_r, p_p = stats.pearsonr(x, y)
    s_r, s_p = stats.spearmanr(x, y)
    
    return {
        'pearson_r': float(p_r),
        'pearson_p': float(p_p),
        'spearman_r': float(s_r),
        'spearman_p': float(s_p),
        'label': label
    }


def exploratory_contrast(a, b, label: str = '', n_permutations: int = 1000, seed: int = 42):
    """
    Exploratory (uncorrected) contrast between two groups.

    Wraps paired_permutation_test without Holm-Bonferroni correction.
    Results are labelled as exploratory and must not be used for
    confirmatory inference without further correction.

    Parameters
    ----------
    a, b : list or array-like of floats
        Paired observations for the two conditions.
    label : str
        Description of the contrast (for reporting purposes).
    n_permutations : int
        Number of permutations for the p-value estimate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        mean_delta      : float  -- mean(a) - mean(b)
        p_value         : float  -- uncorrected permutation p-value
        cohens_d        : float  -- Cohen's d effect size
        label           : str
        note            : str    -- reminder that this is exploratory/uncorrected
    """
    mean_delta, p_value = paired_permutation_test(a, b, n_permutations=n_permutations, seed=seed)
    d = cohens_d(a, b)
    return {
        'mean_delta': mean_delta,
        'p_value': p_value,
        'cohens_d': d,
        'label': label,
        'note': 'EXPLORATORY: uncorrected p-value; do not use for confirmatory inference without Holm correction',
    }

"""
Statistical analysis utilities for experiment results.
"""

import numpy as np
from scipy import stats


def compute_statistics(losses, method='mean_std'):
    """
    Compute statistics for a list of losses.
    
    Args:
        losses: list of loss values
        method: 'mean_std' for mean/std, 'median_iqr' for median/IQR
        
    Returns:
        dictionary with statistics
    """
    clean_losses = [l for l in losses if not np.isnan(l)]
    
    if not clean_losses:
        return {'mean': np.nan, 'std': np.nan, 'raw': losses}
    
    if method == 'mean_std':
        return {
            'mean': float(np.mean(clean_losses)),
            'std': float(np.std(clean_losses)),
            'raw': clean_losses
        }
    elif method == 'median_iqr':
        median = np.nanmedian(clean_losses)
        q75, q25 = np.nanpercentile(clean_losses, [75, 25])
        iqr = q75 - q25
        return {
            'median': float(median),
            'iqr': float(iqr),
            'raw': clean_losses,
            'nan_count': len(losses) - len(clean_losses)
        }
    else:
        raise ValueError(f"Unknown method: {method}")


def t_test(group1, group2):
    """
    Perform t-test between two groups.
    
    Args:
        group1: list of values for first group
        group2: list of values for second group
        
    Returns:
        t_statistic, p_value
    """
    clean1 = [g for g in group1 if not np.isnan(g)]
    clean2 = [g for g in group2 if not np.isnan(g)]
    
    if len(clean1) < 2 or len(clean2) < 2:
        return np.nan, np.nan
    
    t_stat, p_value = stats.ttest_ind(clean1, clean2)
    return t_stat, p_value


def wilcoxon_test(group1, group2):
    """
    Perform Wilcoxon signed-rank test between two groups.
    
    Args:
        group1: list of values for first group
        group2: list of values for second group
        
    Returns:
        statistic, p_value
    """
    clean1 = [g for g in group1 if not np.isnan(g)]
    clean2 = [g for g in group2 if not np.isnan(g)]
    
    if len(clean1) < 2 or len(clean2) < 2:
        return np.nan, np.nan
    
    # Ensure same length for paired test
    min_len = min(len(clean1), len(clean2))
    stat, p_value = stats.wilcoxon(clean1[:min_len], clean2[:min_len])
    return stat, p_value

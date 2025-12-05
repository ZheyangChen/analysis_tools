import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp, poisson
from typing import Optional, Tuple, Dict, Any


def compare_total_rate(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    weight1: Optional[str] = None,
    weight2: Optional[str] = "weight",
    label1: str = "Set1",
    label2: str = "Set2"
) -> Dict[str, Any]:
    """
    Compare total event rates between two datasets (e.g. data vs MC).

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        Input DataFrames.
    weight1, weight2 : str or None
        Column names for weights. If None, assumes weight=1.
    label1, label2 : str
        Optional labels for reporting.

    Returns
    -------
    dict
        Dictionary with total rates, difference, and statistical significance.
    """
    w1 = df1[weight1] if weight1 else np.ones(len(df1))
    w2 = df2[weight2] if weight2 else np.ones(len(df2))

    total1 = w1.sum()
    total2 = w2.sum()
    
    err1 = np.sqrt(np.sum(w1**2))
    err2 = np.sqrt(np.sum(w2**2))

    diff = total1 - total2
    frac_diff = diff / total2 if total2 > 0 else np.nan
    err_combined = np.sqrt(err1**2 + err2**2)
    significance = diff / err_combined if err_combined > 0 else np.nan

    result = {
        f"{label1}_rate": total1,
        f"{label2}_rate": total2,
        "difference": diff,
        "fractional_difference": frac_diff,
        "stat_error_combined": err_combined,
        "significance": significance,
    }

    print("--- Total Rate Comparison ---")
    for k, v in result.items():
        print(f"{k:>25}: {v:.4f}")

    return result


def chi2_binned(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    variable: str,
    bins=20,
    range=None,
    weight1: Optional[str] = None,
    weight2: Optional[str] = "weight"
) -> Tuple[float, float]:
    """
    Perform chi-square test between two weighted distributions.

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        Input DataFrames.
    variable : str
        Column name to compare.
    bins : int or sequence
        Binning specification.
    range : tuple, optional
        Bin range.
    weight1, weight2 : str or None
        Column names for weights. If None, weight = 1.

    Returns
    -------
    chi2 : float
    p_value : float
    """
    x1 = df1[variable]
    x2 = df2[variable]
    w1 = df1[weight1] if weight1 else np.ones(len(df1))
    w2 = df2[weight2] if weight2 else np.ones(len(df2))

    hist1, bins = np.histogram(x1, bins=bins, range=range, weights=w1)
    hist2, _    = np.histogram(x2, bins=bins, range=range, weights=w2)

    # Avoid division by zero
    mask = (hist2 > 0)

    if np.sum(mask) < 2:
        raise ValueError("Too few non-empty bins for chi-square test.")

    expected_scaled = hist2[mask] * (np.sum(hist1[mask]) / np.sum(hist2[mask]))

    chi2_stat = np.sum((hist1[mask] - expected_scaled)**2 / expected_scaled)
    dof = np.sum(mask) - 1
    p_val = 1 - chisquare(hist1[mask], expected_scaled)[1]  # Now this should not raise an error    
        

    print("--- Chi-square Test ---")
    print(f"Chi² statistic: {chi2_stat:.2f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_val:.4f}")

    return chi2_stat, p_val


def ks_test(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    variable: str
) -> Tuple[float, float]:
    """
    Unweighted KS test between two samples.
    This version assumes unweighted (or equally weighted) input.
    """
    x1 = df1[variable]
    x2 = df2[variable]
    ks_stat, p_val = ks_2samp(x1, x2)

    print("--- KS Test ---")
    print(f"KS statistic: {ks_stat:.4f}")
    print(f"p-value: {p_val:.4f}")

    return ks_stat, p_val



def poisson_binned(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    variable: str,
    bins=20,
    range=None,
    weight1: Optional[str] = None,
    weight2: Optional[str] = "weight"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Poisson likelihood test comparing counts in df1 vs expected in df2.

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        Input DataFrames. Typically df1 = data, df2 = MC.
    variable : str
        Column name to compare.
    bins : int or sequence
        Binning specification.
    range : tuple, optional
        Bin range.
    weight1, weight2 : str or None
        Column names for weights. If None, weight = 1.

    Returns
    -------
    observed : np.ndarray
        Counts from df1.
    expected : np.ndarray
        Expected counts from df2 (MC).
    """
    x1 = df1[variable]
    x2 = df2[variable]
    w1 = df1[weight1] if weight1 else np.ones(len(df1))
    w2 = df2[weight2] if weight2 else np.ones(len(df2))

    observed, bins = np.histogram(x1, bins=bins, range=range, weights=w1)
    expected, _    = np.histogram(x2, bins=bins, range=range, weights=w2)

    print("--- Poisson Test (Binned) ---")
    for i in range(len(observed)):
        obs = observed[i]
        exp = expected[i]

        if exp <= 0:
            print(f"Bin {i}: Expected = 0 → skipping")
            continue

        # two-sided p-value
        k = int(np.round(obs))
        p_obs = poisson.pmf(k, exp)
        possible_ks = np.arange(0, int(max(30, k + 10 * np.sqrt(exp))))
        p_vals = poisson.pmf(possible_ks, exp)
        p = np.sum(p_vals[p_vals <= p_obs])
        print(f"Bin {i}: Data = {obs:.2f}, MC = {exp:.2f}, p = {p:.4f}")

    return observed, expected
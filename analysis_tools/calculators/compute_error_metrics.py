import numpy as np
import pandas as pd

def compute_error_metrics(
    df: pd.DataFrame,
    reco_col: str,
    true_col: str,
    weight_col: str = None,
    error_mode: str = "absolute"
) -> dict:
    """
    Compute the key resolution metrics for one reco vs true variable.

    Parameters
    ----------
    df : pandas.DataFrame
    reco_col : str
        Name of the reconstructed‐value column.
    true_col : str
        Name of the true‐value column.
    weight_col : str, optional
        Name of the weight column.  If None, all weights=1.
    error_mode : {'absolute','signed','relative'}
        - 'absolute':  |reco−true|
        - 'signed':    (reco−true)
        - 'relative':  (reco−true)/true

    Returns
    -------
    metrics : dict with keys
       'mean', 'std', 'rms', 'mad', 'iqr', 'p16', 'p84', 'width68'
    """
    # 1) drop any NaNs in the three columns
    sub = df[[reco_col, true_col]].dropna()
    r = sub[reco_col].values
    t = sub[true_col].values

    # 2) build the error array according to mode
    if error_mode == "absolute":
        errs = np.abs(r - t)
    elif error_mode == "signed":
        errs = r - t
    elif error_mode == "relative":
        mask = t != 0
        errs = (r[mask] - t[mask]) / t[mask]
    else:
        raise ValueError("error_mode must be 'absolute','signed', or 'relative'")

    # 3) get weights
    if weight_col and weight_col in df.columns:
        w = df.loc[sub.index, weight_col].values
    else:
        w = np.ones_like(errs, float)

    # 4) compute metrics
    mean = np.average(errs, weights=w)
    var  = np.average((errs - mean)**2, weights=w)
    std  = np.sqrt(var)
    rms  = np.sqrt(np.average(errs**2, weights=w))

    # robust metrics
    # (we ignore weights for percentiles & median‐MAD—IQR, but you could weight them too)
    p16, p50, p84 = np.percentile(errs, [16,50,84])
    mad   = np.median(np.abs(errs - p50))
    iqr   = np.percentile(errs, 75) - np.percentile(errs, 25)
    width68 = p84 - p16

    return {
        "mean":    mean,
        "std":     std,
        "rms":     rms,
        "mad":     mad,
        "iqr":     iqr,
        "p16":     p16,
        "p84":     p84,
        "width68": width68
    }
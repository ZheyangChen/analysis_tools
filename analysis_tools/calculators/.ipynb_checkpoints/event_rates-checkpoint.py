import numpy as np
import pandas as pd

def compute_rate(df: pd.DataFrame,
                 weight_column: str,
                 print_raw: bool = False):
    """
    Compute (and optionally print) the raw count, weighted rate, and its uncertainty
    for a single weight column.
    """
    clean = df.dropna(subset=[weight_column])
    raw_count = len(clean)
    w = clean[weight_column].values

    rate = w.sum()
    uncertainty = np.sqrt((w**2).sum())

    if print_raw:
        print(f"Raw event count: {raw_count}")

    # Always print rate & uncertainty:
    print(f"[{weight_column}] rate        = {rate:.6g}")
    print(f"[{weight_column}] uncertainty = {uncertainty:.6g}")

    return raw_count, rate, uncertainty


def compute_common_rates(df: pd.DataFrame,
                         weight_columns=None,
                         print_raw: bool = True):
    """
    Compute rates & uncertainties for a set of “common” weight columns—and any extras.

    This will print the raw event count once (if requested), then for each weight
    column print its rate and uncertainty.

    Returns a dict mapping weight_column -> (raw_count, rate, unc).
    """
    defaults = ['weight', 'astro_weight', 'conv_weight', 'prompt_weight']
    if weight_columns is None:
        to_do = defaults
    else:
        extras = [w for w in weight_columns if w not in defaults]
        to_do = defaults + extras

    results = {}

    # Print raw count once, based on the first weight in to_do
    if print_raw and to_do:
        raw_all = len(df.dropna(subset=[to_do[0]]))
        print(f"Raw event count: {raw_all}")

    # Now compute each, but suppress per-weight raw prints
    for wcol in to_do:
        raw, rate, unc = compute_rate(df, wcol, print_raw=False)
        results[wcol] = (raw, rate, unc)

    return results
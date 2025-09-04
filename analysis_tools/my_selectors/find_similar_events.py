import pandas as pd
import numpy as np


def find_similar_events(
    df,
    reference_event,
    features,
    deltas=None,
    mode: str = "absolute",   # "absolute" or "factor"
    factor: float = 2.0
):
    """
    Find events similar to a reference event.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to search.
    reference_event : pd.Series or dict
        The event to compare against.
    features : list of str
        Feature names to match.
    deltas : list or dict, optional
        If mode='absolute': range for each feature (±delta).
    mode : str
        'absolute' for ±delta match, 'factor' for within [x/factor, x*factor].
    factor : float
        If mode='factor': the matching factor (default 2.0).

    Returns
    -------
    pd.DataFrame
        Subset of similar events.
    """
    mask = pd.Series(True, index=df.index)

    for i, feature in enumerate(features):
        ref_val = reference_event[feature].values[0]  # safe scalar

        if mode == "absolute":
            delta = deltas[i] if isinstance(deltas, list) else deltas[feature]
            lower = ref_val - delta
            upper = ref_val + delta
        elif mode == "factor":
            if ref_val == 0:
                lower, upper = -factor, factor
            else:
                lower = ref_val / factor
                upper = ref_val * factor
                if lower > upper:
                    lower, upper = upper, lower
        else:
            raise ValueError(f"Invalid mode: {mode}")

        mask &= (df[feature] >= lower) & (df[feature] <= upper)

    return df[mask]
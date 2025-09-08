import numpy as np
import pandas as pd


def generate_bins(bins_config):
    """
    Generate bins based on configuration.

    Parameters
    ----------
    bins_config : list or dict
        - If it is a list, return it as is.
        - If it is a dict, expect keys: 'function' (either "linspace" or "logspace"),
          'start', 'stop', and 'num'. Generates bins using np.linspace or np.logspace.

    Returns
    -------
    bins : array-like
        The bin edges.
    """
    if isinstance(bins_config, list):
        return bins_config
    elif isinstance(bins_config, dict):
        fn = bins_config.get('function')
        start = bins_config.get('start')
        stop = bins_config.get('stop')
        num = bins_config.get('num')
        if fn == 'linspace':
            return np.linspace(start, stop, num)
        elif fn == 'logspace':
            return np.logspace(start, stop, num)
        else:
            raise ValueError("Unknown bin generation function specified. Use 'linspace' or 'logspace'.")
    else:
        raise ValueError("bins configuration must be either a list or a dict.")
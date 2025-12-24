
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Sequence

def plot_errorbars(
    df: pd.DataFrame,
    features: Union[str, Sequence[str]],
    weight_col: str = None,
    bins: Union[int, Sequence[float]] = 50,
    normalized: bool = False,
    ax: plt.Axes = None,
    labels: Sequence[str] = None,
    colors: Union[str, Sequence[str]] = None,
    fmt: str = 'o',
    capsize: float = 3,
    show: bool = True,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    legend_loc: str = 'best',
    **kwargs
):
    """
    Plot error bars (±√Σw²) for one or more features from the same DataFrame.
    Each feature gets its own color/label if provided.
    """

    # Normalize input
    if isinstance(features, str):
        features = [features]
    n = len(features)

    if labels is None:
        labels = features
    if isinstance(colors, str):
        colors = [colors] * n
    elif colors is None:
        colors = [None] * n

    # Setup plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for i, feature in enumerate(features):
        x = df[feature]
        w = df[weight_col] if weight_col else None

        # Histogram and uncertainties
        counts, edges = np.histogram(x, bins=bins, weights=w, density=normalized)
        sumw2, _ = np.histogram(x, bins=bins, weights=w**2 if weight_col else None)
        errs = np.sqrt(sumw2)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Plot
        mask = counts > 0
        
        ax.errorbar(
            centers[mask], counts[mask],
            yerr=errs[mask],
            fmt=fmt,
            label=labels[i],
            color=colors[i],
            capsize=capsize,
            **kwargs
        )

    # Axes formatting
    ax.set_xlabel(xlabel or "Value")
    ax.set_ylabel(ylabel or ("Normalized Rate" if normalized else "Event Count"))
    if title:
        ax.set_title(title)
    if labels:
        ax.legend(loc=legend_loc)

    if show:
        plt.show()

    return fig, ax
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.gridspec as gridspec
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence, Union, Dict, Any


def plot_histograms(
    data_input: Union[pd.DataFrame, Sequence[pd.DataFrame], Dict[str,pd.DataFrame]],
    plotvar: Union[str, Sequence[str]],
    bins,
    weights_map: Union[str, Dict[str,str]] = 'weight',
    normalized: bool = False,
    histtype: str = 'step',
    xscale: str = None,
    yscale: str = 'log',
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    legend_loc: str = 'best',
    xlim: tuple = None,
    ylim: tuple = None,
    colors: Union[str, Sequence[str]] = None,
    labels: Sequence[str] = None,

    # << NEW >>
    ax: plt.Axes = None,
    vlines: Sequence[float] = None,
    hlines: Sequence[float] = None,
    vline_kwargs: Dict[str,Any] = None,
    hline_kwargs: Dict[str,Any] = None,
    show: bool = True,
    save_path: str = None
) -> plt.Axes:
    """
    Two modes:
      • Multi-DF mode: data_input=dict/list of DataFrames, plotvar=str
      • Multi-column mode: data_input=DataFrame, plotvar=list of str

    New features:
      • ax=...: plot into an existing Axes (returns that Axes)
      • vlines / hlines: list of x/y positions to draw axvline()/axhline()
      • vline_kwargs / hline_kwargs: kwargs passed to those calls
    """
    # — Figure out data & columns (unchanged) —
    if isinstance(data_input, dict) or isinstance(data_input, list):
        if isinstance(data_input, list):
            data_dict = {f'df_{i}': df for i, df in enumerate(data_input)}
        else:
            data_dict = data_input
        if isinstance(plotvar, (list, tuple)):
            raise ValueError("When data_input is dict/list, plotvar must be a single column name.")
        dfs      = [data_dict[k] for k in data_dict]
        plotvars = [plotvar] * len(dfs)
        if labels is None:
            labels = list(data_dict.keys())

    elif isinstance(data_input, pd.DataFrame):
        if isinstance(plotvar, (list, tuple)):
            dfs      = [data_input] * len(plotvar)
            plotvars = list(plotvar)
            if labels is None:
                labels = plotvar
        else:
            dfs      = [data_input]
            plotvars = [plotvar]
            if labels is None:
                labels = [plotvar]
    else:
        raise ValueError("data_input must be a DataFrame, list or dict of DataFrames")

    n = len(dfs)

    # — Normalize colors & labels —
    if colors is None:
        colors = [None]*n
    elif isinstance(colors, str):
        colors = [colors]*n
    elif len(colors) != n:
        raise ValueError(f"{n} histograms but {len(colors)} colors")

    if labels is None or len(labels) != n:
        raise ValueError(f"Must provide exactly {n} labels")

    # — Set up Axes —
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # — Plot histograms —
    for df, var, col, lab in zip(dfs, plotvars, colors, labels):
        if isinstance(weights_map, str):
            w = df[weights_map]
        else:
            wspec = weights_map.get(lab)
            if wspec is None:
                raise ValueError(f"No weight for label {lab!r}")
            w = df[wspec].sum(axis=1) if isinstance(wspec, list) else df[wspec]

        ax.hist(
            df[var],
            bins=bins,
            histtype=histtype,
            weights=w,
            density=normalized,
            color=col,
            label=lab
        )

    # — Scales & limits —
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if xlim:   ax.set_xlim(xlim)
    if ylim:   ax.set_ylim(ylim)

    # — Labels & legend —
    is_multi = len(plotvars) > 1
    ax.set_xlabel(xlabel or ("" if is_multi else plotvars[0]))
    default_ylabel = 'Probability Density' if normalized else 'Rate per Year'
    ax.set_ylabel(ylabel or default_ylabel)
    ax.set_title(title or "")
    ax.legend(loc=legend_loc)

    # — Optional v/h lines —
    vline_kwargs = vline_kwargs or {}
    hline_kwargs = hline_kwargs or {}
    if vlines:
        for x in vlines:
            ax.axvline(x, **vline_kwargs)
    if hlines:
        for y in hlines:
            ax.axhline(y, **hline_kwargs)

    # — Save / Show —
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()

    return ax


def compute_hist_with_errors(data, bins, weight, transform_func=None):
    """
    Compute histogram counts and uncertainties.
    
    Parameters:
    - data: array-like, the data to histogram.
    - bins: array-like, the bin edges.
    - weight: array-like, weights for each data point.
    - transform_func: function or None, an optional function to transform the data (e.g., np.log10).
    
    Returns:
    - hist: array, weighted counts per bin.
    - bin_edges: array, the bin edges.
    - error: array, the uncertainties (sqrt(sum(weight^2)) per bin.
    """
    if transform_func is not None:
        data = transform_func(data)
    hist, bin_edges = np.histogram(data, bins=bins, weights=weight)
    sumw2, _ = np.histogram(data, bins=bins, weights=weight**2)
    error = np.sqrt(sumw2)
    return hist, bin_edges, error

def plot_stacked_hist_with_ratio(hist_data, errorbar_data, plotvar, bins,
                                 hist_weight, errorbar_weight, transform_func=None,
                                 xscale=None, yscale='log',
                                 xlabel=None, ylabel_hist='Rate per Year', ylabel_ratio='Ratio',
                                 title=None, legend_loc='best',
                                 xlim=None, ylim_hist=None, ylim_ratio=None,
                                 colors=None, errorbar_label='Data', ratio_label='Ratio (MC/Data)'):
    # Convert hist_data to dict if needed.
    if isinstance(hist_data, pd.DataFrame):
        hist_data = {'data': hist_data}
    elif isinstance(hist_data, list):
        hist_data = {f'df_{i}': df for i, df in enumerate(hist_data)}
    
    # Create figure and manually position axes for perfect alignment.
    fig = plt.figure(figsize=(8, 6))
    ax_hist = fig.add_axes([0.1, 0.3, 0.85, 0.6])
    ax_ratio = fig.add_axes([0.1, 0.1, 0.85, 0.2], sharex=ax_hist)
    
    # ----- Stacked Histogram -----
    stacked_datasets, stacked_weights, labels = [], [], []
    for label, df in hist_data.items():
        data = df[plotvar]
        if transform_func is not None:
            data = transform_func(data)
        stacked_datasets.append(data)
        stacked_weights.append(df[hist_weight])
        labels.append(label)
    
    ax_hist.hist(stacked_datasets, bins=bins, weights=stacked_weights,
                 stacked=True, label=labels, histtype='bar', alpha=0.5, color=colors)
    
    # ----- Errorbar Histogram -----
    if isinstance(errorbar_data, dict):
        errorbar_df = errorbar_data.get('data', list(errorbar_data.values())[0])
    else:
        errorbar_df = errorbar_data
    
    data_arr = errorbar_df[plotvar]
    hist_errorbar, bin_edges, error = compute_hist_with_errors(
        data_arr, bins, errorbar_df[errorbar_weight], transform_func=transform_func)
    
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax_hist.errorbar(bin_centers, hist_errorbar, yerr=error, fmt='k.', elinewidth=1, ms=3,
                     label=errorbar_label)
    
    # ----- Ratio Calculation -----
    combined_hist = np.sum([np.histogram(
        transform_func(df[plotvar]) if transform_func else df[plotvar],
        bins=bins, weights=df[hist_weight])[0] for df in hist_data.values()], axis=0)
    combined_error = np.sqrt(np.sum([np.histogram(
        transform_func(df[plotvar]) if transform_func else df[plotvar],
        bins=bins, weights=df[hist_weight]**2)[0] for df in hist_data.values()], axis=0))
    
    mask = hist_errorbar > 0
    ratio = np.full_like(combined_hist, np.nan, dtype=float)
    ratio_error = np.full_like(combined_hist, np.nan, dtype=float)
    ratio[mask] = combined_hist[mask] / hist_errorbar[mask]
    rel_error_combined = np.zeros_like(combined_hist, dtype=float)
    rel_error_errorbar = np.zeros_like(hist_errorbar, dtype=float)
    rel_error_combined[mask] = combined_error[mask] / combined_hist[mask]
    rel_error_errorbar[mask] = error[mask] / hist_errorbar[mask]
    ratio_error[mask] = ratio[mask] * np.sqrt(rel_error_combined[mask]**2 + rel_error_errorbar[mask]**2)
    
    ax_ratio.step(bin_centers, ratio, where='mid', label=ratio_label)
    ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_error, fmt='k.', elinewidth=1, ms=3)
    ax_ratio.axhline(1, color='black', linestyle='--', linewidth=1)
    
    # ----- Axis Scales, Limits, and Labels -----
    if xscale:
        ax_hist.set_xscale(xscale)
        ax_ratio.set_xscale(xscale)
    if yscale:
        ax_hist.set_yscale(yscale)
    if xlim:
        ax_hist.set_xlim(xlim)
        ax_ratio.set_xlim(xlim)
    if ylim_hist:
        ax_hist.set_ylim(ylim_hist)
    if ylim_ratio:
        ax_ratio.set_ylim(ylim_ratio)
    
    ax_hist.tick_params(labelbottom=False)
    ax_ratio.set_xlabel(xlabel if xlabel else plotvar)
    ax_hist.set_ylabel(ylabel_hist)
    ax_ratio.set_ylabel(ylabel_ratio)
    
    if title:
        ax_hist.set_title(title)
    
    ax_hist.legend(loc=legend_loc)
    ax_ratio.legend(loc=legend_loc)
    
    plt.tight_layout()
    plt.show()
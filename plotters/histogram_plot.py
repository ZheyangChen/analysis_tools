import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.gridspec as gridspec
from math import sqrt



def plot_histograms(data_input, plotvar, bins, weights_map='weight', normalized=False,
                    histtype='step', xscale=None, yscale='log',
                    xlabel=None, ylabel=None, title=None,
                    legend_loc='best', show=True, save_path=None):
    """
    Plot histograms for one or multiple datasets on a single figure.
    
    Parameters:
    -----------
    data_input : dict, list, or pd.DataFrame
        - If dict: keys are labels and values are pandas DataFrames.
        - If list: a list of pandas DataFrames; labels will be auto-generated.
        - If a single pd.DataFrame: it will be plotted with a default label.
    plotvar : str
        Column name in each DataFrame to be plotted.
    bins : array-like
        Bin edges or binning configuration (e.g., np.linspace(...)).
    weights_map : dict or str
        - If dict: mapping each label (or auto-generated label) to its weight specification.
          The weight specification can be:
            - A single string (e.g. 'astro_weight') or
            - A list of strings (e.g. ['astro_weight', 'conv_weight']) to sum them.
        - If str: the same weight column name is used for all DataFrames.
    normalized : bool, default False
        If True, plot the histogram as a normalized (density) histogram.
    histtype : str, default 'step'
        Type of histogram to draw.
    xscale : str or None
        Scale of the x-axis ('log', 'linear', etc.). If None, uses default.
    yscale : str or None
        Scale of the y-axis. Default is 'log'.
    xlabel : str or None
        Label for the x-axis. If None, defaults to plotvar.
    ylabel : str or None
        Label for the y-axis. If None, defaults to 'Rate per Year' unless normalized,
        in which case it might be 'Probability Density'.
    title : str or None
        Plot title. If None, defaults to plotvar.
    legend_loc : str, default 'best'
        Location of the legend.
    show : bool, default True
        Whether to display the plot.
    save_path : str or None
        If provided, the plot will be saved to this path.
    """
    # Convert data_input to a dictionary if necessary.
    if isinstance(data_input, pd.DataFrame):
        data_dict = {'data': data_input}
    elif isinstance(data_input, list):
        data_dict = {f'df_{i}': df for i, df in enumerate(data_input)}
    elif isinstance(data_input, dict):
        data_dict = data_input
    else:
        raise ValueError("data_input must be a DataFrame, a list of DataFrames, or a dict of DataFrames.")
    
    plt.figure()
    
    for label, df in data_dict.items():
        # Determine weights: if a list is provided, sum the columns row-wise.
        if isinstance(weights_map, str):
            weights_spec = weights_map
        else:
            weights_spec = weights_map.get(label)
            if weights_spec is None:
                raise ValueError(f"No weight specification found for label '{label}'.")
        
        if isinstance(weights_spec, list):
            weights = df[weights_spec].sum(axis=1)
        else:
            weights = df[weights_spec]
        
        plt.hist(df[plotvar], bins=bins, histtype=histtype,
                 weights=weights, density=normalized, label=label)
    
    if xscale:
        plt.xscale(xscale)
    if yscale:
        plt.yscale(yscale)
    
    plt.xlabel(xlabel if xlabel else plotvar)
    # Adjust y-axis label based on normalized flag.
    default_ylabel = 'Probability Density' if normalized else 'Rate per Year'
    plt.ylabel(ylabel if ylabel else default_ylabel)
    plt.title(title if title else plotvar)
    plt.legend(loc=legend_loc)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()
        
        

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
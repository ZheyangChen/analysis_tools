import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from analysis_tools.plotters.histogram_plot import plot_histograms, plot_stacked_hist_with_ratio
from analysis_tools.utils.utils import generate_bins


import os


def plot_multi_var(df, config, errorbar_data=None):
    """
    Generate plots based on a YAML configuration and a given DataFrame.

    If common.errorbar=True in config, all histogram plots will show error bars.
    For 'stacked_hist_with_ratio' plots, errorbar_data (a dict of DataFrames) can be passed in.

    Parameters
    ----------
    df : pandas.DataFrame
        The primary DataFrame containing your data.
    config : dict
        The plotting configuration dictionary (loaded from a YAML file).
    errorbar_data : dict or None
        Data for errorbars in stacked‐hist plots; if None, df is used.
    """
    results = {}

    # default errorbar_data for stacked plots
    if errorbar_data is None:
        errorbar_data = {"data": df}

    common = config.get('common', {})
    prefix           = common.get('prefix', "")
    common_xscale    = common.get('xscale')
    common_yscale    = common.get('yscale')
    common_xlabel    = common.get('xlabel')
    common_ylabel    = common.get('ylabel')
    common_title     = common.get('title')
    common_weights   = common.get('weights_map', 'weight')
    common_normalized= common.get('normalized', False)
    common_histtype  = common.get('histtype', 'step')
    common_legend    = common.get('legend_loc', 'best')
    common_save_path = common.get('save_path')
    common_show      = common.get('show', True)
    common_errorbar  = common.get('errorbar', False)    # ← new!
    common_colors    = common.get('colors')             # list of colors

    common_hist_weight = common.get('hist_weight', common_weights)
    common_err_weight  = common.get('errorbar_weight', common_weights)

    plot_type   = config.get('plot_type', 'histogram')
    var_configs = config.get('variables', {})

    if plot_type == 'histogram':
        for i, (var_key, var_cfg) in enumerate(var_configs.items()):
            col_tmpl  = var_cfg.get('column', var_key)
            column    = col_tmpl.format(prefix=prefix)
            bins_cfg  = var_cfg.get('bins')
            bins      = generate_bins(bins_cfg)

            weights_map = var_cfg.get('weights_map', common_weights)
            normalized  = var_cfg.get('normalized', common_normalized)
            histtype    = var_cfg.get('histtype', common_histtype)
            xlabel      = var_cfg.get('xlabel', common_xlabel or column)
            ylabel      = var_cfg.get('ylabel', common_ylabel)
            xscale      = var_cfg.get('xscale', common_xscale)
            yscale      = var_cfg.get('yscale', common_yscale)
            title       = var_cfg.get('title', common_title or column)
            legend_loc  = var_cfg.get('legend_loc', common_legend)

            # pick per‐variable color if given, else from common_colors list
            color = var_cfg.get('color')
            if not color and isinstance(common_colors, list):
                color = common_colors[i] if i < len(common_colors) else None

            plot_histograms(
                data_input = df,
                plotvar    = column,
                bins       = bins,
                weights_map= weights_map,
                normalized = normalized,
                histtype   = histtype,
                xscale     = xscale,
                yscale     = yscale,
                xlabel     = xlabel,
                ylabel     = ylabel,
                title      = title,
                legend_loc = legend_loc,
                show       = common_show,
                save_path  = common_save_path,
                colors     = color,
                errorbar   = common_errorbar      # ← pass it here
            )
            results[var_key] = None

    elif plot_type == 'stacked_hist_with_ratio':
        for i, (var_key, var_cfg) in enumerate(var_configs.items()):
            col_tmpl    = var_cfg.get('column', var_key)
            column      = col_tmpl.format(prefix=prefix)
            bins_cfg    = var_cfg.get('bins')
            bins        = generate_bins(bins_cfg)

            hist_weight    = var_cfg.get('hist_weight', common_hist_weight)
            errorbar_weight= var_cfg.get('errorbar_weight', common_err_weight)
            transform_func = var_cfg.get('transform_func')  # should already be a callable or None

            xlabel      = var_cfg.get('xlabel', column)
            ylabel_hist = var_cfg.get('ylabel',       'Frequency')
            ylabel_ratio= var_cfg.get('ylabel_ratio','Ratio')
            title       = var_cfg.get('title',        column)
            legend_loc  = var_cfg.get('legend_loc',   common_legend)
            xscale      = var_cfg.get('xscale',       common_xscale)
            yscale      = var_cfg.get('yscale',       common_yscale)
            xlim        = var_cfg.get('xlim')
            ylim_hist   = var_cfg.get('ylim_hist')
            ylim_ratio  = var_cfg.get('ylim_ratio')
            errorbar_lab= var_cfg.get('errorbar_label', 'Data')
            ratio_label = var_cfg.get('ratio_label', 'Ratio (MC/Data)')

            # colors for stacked bars: use common_colors list if provided
            colors = var_cfg.get('color') or common_colors

            plot_stacked_hist_with_ratio(
                hist_data       = df,
                errorbar_data   = errorbar_data,
                plotvar         = column,
                bins            = bins,
                hist_weight     = hist_weight,
                errorbar_weight = errorbar_weight,
                transform_func  = transform_func,
                xscale          = xscale,
                yscale          = yscale,
                xlabel          = xlabel,
                ylabel_hist     = ylabel_hist,
                ylabel_ratio    = ylabel_ratio,
                title           = title,
                legend_loc      = legend_loc,
                xlim            = xlim,
                ylim_hist       = ylim_hist,
                ylim_ratio      = ylim_ratio,
                colors          = colors,
                errorbar_label  = errorbar_lab,
                ratio_label     = ratio_label
            )
            results[var_key] = None

    else:
        raise ValueError(f"Unknown plot_type '{plot_type}' – must be 'histogram' or 'stacked_hist_with_ratio'")

    return results

# Example usage in main:
if __name__ == '__main__':
    # Load a sample DataFrame (replace this with your actual data loading)
    df = pd.DataFrame({
        'A': np.random.randn(1000),         # Data for variable A
        'B': np.random.rand(1000) * 10,       # Data for variable B
        'weight': np.random.rand(1000)        # Weight column
    })
    
    # Load the YAML configuration.
    with open('plot_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Call the multi-plot function: pass in the DataFrame and the configuration dictionary.
    plot_multi_var(df, config)
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from analysis_tools.plotters.histogram_plot import plot_histograms, plot_stacked_hist_with_ratio
from analysis_tools.utils import generate_bins

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from analysis_tools.plotters.histogram_plot import plot_histograms, plot_stacked_hist_with_ratio
from analysis_tools.utils import generate_bins

def plot_multi_var(df, config, errorbar_data=None):
    """
    Generate plots based on a YAML configuration and a given DataFrame.
    
    For standard histogram plots, the primary DataFrame (df) is used.
    For 'stacked_hist_with_ratio' plots, an optional errorbar_data argument can be provided;
    if not provided, the primary DataFrame is used for errorbars.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The primary DataFrame containing your data.
    config : dict
        The plotting configuration dictionary (loaded from a YAML file).
    errorbar_data : dict or None, optional
        A dictionary of DataFrames to be used as the errorbar data source for stacked
        histogram plots. If None, the primary DataFrame is used.
    
    Returns
    -------
    results : dict
        A dictionary mapping each variable key to the corresponding plot.
    """
    results = {}
    
    # If errorbar_data is not provided, default to using the primary DataFrame.
    if errorbar_data is None:
        errorbar_data = {"data": df}
    
    common = config.get('common', {})
    prefix         = common.get('prefix', "")  # Common prefix from YAML
    common_xscale  = common.get('xscale')
    common_yscale  = common.get('yscale')
    common_xlabel  = common.get('xlabel')
    common_ylabel  = common.get('ylabel')
    common_title   = common.get('title')
    common_weights = common.get('weights_map', 'weight')
    common_normalized = common.get('normalized', False)
    common_histtype = common.get('histtype', 'step')
    common_legend  = common.get('legend_loc', 'best')
    common_save_path = common.get('save_path')
    common_show    = common.get('show', True)
    
    common_hist_weight = common.get('hist_weight', common_weights)
    common_err_weight  = common.get('errorbar_weight', common_weights)
    common_colors = common.get('colors')  # Expecting a list like ['b', 'y', 'g']
    
    plot_type = config.get('plot_type', 'histogram')
    var_configs = config.get('variables', {})
    
    if plot_type == 'histogram':
        # Standard histogram plots.
        for i, (var_key, var_config) in enumerate(var_configs.items()):
            column_template = var_config.get('column', var_key)
            column = column_template.format(prefix=prefix)
            bins_config = var_config.get('bins')
            bins = generate_bins(bins_config)
            weights_map = var_config.get('weights_map', common_weights)
            normalized = var_config.get('normalized', common_normalized)
            histtype = var_config.get('histtype', common_histtype)
            xlabel = var_config.get('xlabel', common_xlabel if common_xlabel is not None else column)
            ylabel = var_config.get('ylabel', common_ylabel)
            xscale = var_config.get('xscale', common_xscale)
            yscale = var_config.get('yscale', common_yscale)
            title = var_config.get('title', common_title if common_title is not None else column)
            legend_loc = var_config.get('legend_loc', common_legend)
            # For histogram plots, determine per variable color (as a scalar).
            color = var_config.get('color')
            if not color and common_colors and isinstance(common_colors, list):
                if i < len(common_colors):
                    color = common_colors[i]
                else:
                    color = None
            plot_histograms(
                data_input=df,
                plotvar=column,
                bins=bins,
                weights_map=weights_map,
                normalized=normalized,
                histtype=histtype,
                xscale=xscale,
                yscale=yscale,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                legend_loc=legend_loc,
                show=common_show,
                save_path=common_save_path,
                colors=color
            )
            results[var_key] = None
    elif plot_type == 'stacked_hist_with_ratio':
        # For stacked hist with ratio, we want to pass a color list if available.
        for i, (var_key, var_config) in enumerate(var_configs.items()):
            column_template = var_config.get('column', var_key)
            column = column_template.format(prefix=prefix)
            bins_config = var_config.get('bins')
            bins = generate_bins(bins_config)
            hist_weight = var_config.get('hist_weight', common_hist_weight)
            errorbar_weight = var_config.get('errorbar_weight', common_err_weight)
            transform_func = var_config.get('transform_func')  # Assume callable or None.
            xlabel = var_config.get('xlabel', column)
            ylabel_hist = var_config.get('ylabel', 'Frequency')
            ylabel_ratio = var_config.get('ylabel_ratio', 'Ratio')
            title = var_config.get('title', column)
            legend_loc = var_config.get('legend_loc', 'best')
            xscale = var_config.get('xscale', common_xscale)
            yscale = var_config.get('yscale', common_yscale)
            xlim = var_config.get('xlim')
            ylim_hist = var_config.get('ylim_hist')
            ylim_ratio = var_config.get('ylim_ratio')
            errorbar_label = var_config.get('errorbar_label', 'Data')
            ratio_label = var_config.get('ratio_label', 'Ratio (MC/Data)')
            # For stacked hist with ratio, check for variable-specific color.
            color = var_config.get('color')
            if not color and common_colors and isinstance(common_colors, list):
                # For stacked hist, pass the entire list from common if available.
                color = common_colors
            used_err_data = errorbar_data  # use the provided errorbar_data dictionary.
            plot_stacked_hist_with_ratio(
                hist_data=df,
                errorbar_data=used_err_data,
                plotvar=column,
                bins=bins,
                hist_weight=hist_weight,
                errorbar_weight=errorbar_weight,
                transform_func=transform_func,
                xscale=xscale,
                yscale=yscale,
                xlabel=xlabel,
                ylabel_hist=ylabel_hist,
                ylabel_ratio=ylabel_ratio,
                title=title,
                legend_loc=legend_loc,
                xlim=xlim,
                ylim_hist=ylim_hist,
                ylim_ratio=ylim_ratio,
                colors=color,
                errorbar_label=errorbar_label,
                ratio_label=ratio_label
            )
            results[var_key] = None
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Supported types are 'histogram' and 'stacked_hist_with_ratio'.")
    
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
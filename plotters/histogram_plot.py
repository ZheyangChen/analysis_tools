import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(data_dict, plotvar, bins, weights_map, histtype='step',
                    xscale=None, yscale='log', xlabel=None, ylabel=None, title=None,
                    legend_loc='upper left', show=True, save_path=None):
    """
    Plot histograms for multiple datasets on a single figure.
    
    Parameters:
    - data_dict: dict
        Dictionary of datasets. Keys are labels and values are pandas DataFrames.
    - plotvar: str
        Column name in each DataFrame to be plotted.
    - bins: array-like
        Bin edges or binning configuration (e.g., np.linspace(...)).
    - weights_map: dict
        Dictionary mapping each label to its weight specification.
        The weight specification can be:
            - A single string: a column name in the DataFrame.
            - A list of strings: names of columns to sum up as weights.
    - histtype: str, default 'step'
        Type of histogram to draw.
    - xscale: str or None
        Scale of the x-axis ('log', 'linear', etc.). If None, uses default.
    - yscale: str or None
        Scale of the y-axis. Default is 'log'.
    - xlabel: str or None
        Label for the x-axis. If None, defaults to plotvar.
    - ylabel: str or None
        Label for the y-axis. If None, defaults to 'Rate per Year'.
    - title: str or None
        Plot title. If None, defaults to plotvar.
    - legend_loc: str, default 'upper left'
        Location of the legend.
    - show: bool, default True
        Whether to display the plot.
    - save_path: str or None
        If provided, the plot will be saved to this path.
    """
    plt.figure()
    
    for label, df in data_dict.items():
        # Determine weights: if a list is provided, sum the columns row-wise.
        weights_spec = weights_map.get(label)
        if isinstance(weights_spec, list):
            weights = df[weights_spec].sum(axis=1)
        else:
            weights = df[weights_spec]
        
        plt.hist(df[plotvar], bins=bins, histtype=histtype,
                 weights=weights, label=label)
    
    if xscale:
        plt.xscale(xscale)
    if yscale:
        plt.yscale(yscale)
    
    plt.xlabel(xlabel if xlabel else plotvar)
    plt.ylabel(ylabel if ylabel else 'Rate per Year')
    plt.title(title if title else plotvar)
    plt.legend(loc=legend_loc)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

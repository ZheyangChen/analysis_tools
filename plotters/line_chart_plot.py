import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_line_chart(x, y, y_err=None, marker="o", linestyle="-", color=None, label=None,
                    xscale=None, yscale=None, xlim=None, ylim=None):
    """
    Plot a general-purpose line chart with optional error bands and axis customizations.
    
    Parameters
    ----------
    x : array-like
        x-axis data.
    y : array-like
        y-axis data.
    y_err : array-like or None, optional
        Error values for y. If provided, error bands are plotted.
    marker : str, optional
        Marker style.
    linestyle : str, optional
        Line style.
    color : str or None, optional
        Color for the plot. If None, Matplotlib's default color cycle is used.
    label : str, optional
        Label for the plot (for the legend).
    xscale : str or None, optional
        Scale for the x-axis (e.g., 'linear', 'log').
    yscale : str or None, optional
        Scale for the y-axis.
    xlim : tuple or None, optional
        Limits for the x-axis as (min, max).
    ylim : tuple or None, optional
        Limits for the y-axis as (min, max).
    
    Returns
    -------
    fig, ax : tuple
        The matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots()
    
    # Plot the line; let Matplotlib choose a color if none is provided.
    if color is None:
        lines = ax.plot(x, y, marker=marker, linestyle=linestyle, label=label)
        actual_color = lines[0].get_color()
    else:
        ax.plot(x, y, marker=marker, linestyle=linestyle, color=color, label=label)
        actual_color = color
        
    if y_err is not None:
        ax.fill_between(x, y - y_err, y + y_err, color=actual_color, alpha=0.3)
    
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if label:
        ax.legend()
        
    return fig, ax

def weighted_stats(values, weights):
    """
    Compute the weighted mean and standard deviation.

    Parameters
    ----------
    values : array-like
        Data values.
    weights : array-like
        Weights for each data point.

    Returns
    -------
    tuple
        Weighted mean and weighted standard deviation.
    """
    weighted_mean = np.average(values, weights=weights)
    variance = np.average((values - weighted_mean) ** 2, weights=weights)
    return weighted_mean, np.sqrt(variance)

def plot_difference(df, x_value_name, y_value_reco, y_value_true, 
                                y_value_name="difference", weight_name="weight",
                                label="Difference", color=None,
                                bins=None,
                                xscale=None, yscale=None, xlim=None, ylim=None,
                                show=True, save_path=None):
    """
    Specialized function to plot the weighted difference between two y-values,
    with customizable binning and axis options.
    
    The function computes the absolute difference between y_value_reco and y_value_true,
    bins the data along x_value_name (using either pre-specified bins or a default),
    and calculates the weighted mean and standard deviation within each bin.
    It then calls the general-purpose line chart function to plot the results with error bands.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    x_value_name : str
        Column name for x-axis values (used for binning).
    y_value_reco : str
        Column name for the reconstructed y-values.
    y_value_true : str
        Column name for the true y-values.
    y_value_name : str, optional
        Column name for the computed difference (defaults to "difference").
    weight_name : str, optional
        Column name for weights (defaults to "weight").
    label : str, optional
        Label for the plot (defaults to "Difference").
    color : str or None, optional
        Color for the plot. If None, Matplotlib's default color is used.
    bins : array-like or None, optional
        Pre-specified bins for the x-axis. If None, defaults to np.logspace(2, 8, 20).
    xscale : str or None, optional
        Scale for the x-axis (e.g., 'linear', 'log').
    yscale : str or None, optional
        Scale for the y-axis.
    xlim : tuple or None, optional
        Limits for the x-axis as (min, max).
    ylim : tuple or None, optional
        Limits for the y-axis as (min, max).
    show : bool, default True
        Whether to display the plot.
    save_path : str or None, optional
        If provided, the plot is saved to this path.
    
    Returns
    -------
    fig, ax : tuple
        The matplotlib figure and axes objects.
    """
    # Work on a copy of the DataFrame.
    df_copy = df.copy()
    
    # Compute the absolute difference.
    df_copy[y_value_name] = np.abs(df_copy[y_value_reco] - df_copy[y_value_true])
    
    # Use provided bins, or compute default bins.
    if bins is None:
        bins = np.logspace(2, 8, 20)
    
    # Bin the x values.
    df_copy['x_bin'] = pd.cut(df_copy[x_value_name], bins=bins, labels=False)
    
    # Compute weighted statistics per bin.
    results = df_copy.groupby("x_bin").apply(
        lambda group: weighted_stats(group[y_value_name], group[weight_name])
    ).reindex(range(len(bins) - 1), fill_value=(np.nan, np.nan))
    
    weighted_means = results.map(lambda x: x[0])
    weighted_std_devs = results.map(lambda x: x[1])
    valid = ~np.isnan(weighted_means)
    
    # Calculate bin centers.
    x_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot using the general-purpose line chart function.
    fig, ax = plot_line_chart(x_centers[valid], weighted_means[valid],
                              y_err=weighted_std_devs[valid],
                              marker="o", linestyle="-", color=color, label=label,
                              xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim)
    
    ax.set_xlabel(x_value_name)
    ax.set_ylabel(y_value_name)
    ax.set_title("Difference Plot")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

# Example usage:
if __name__ == '__main__':
    np.random.seed(0)
    N = 1000
    df_example = pd.DataFrame({
        'energy': np.random.uniform(100, 1000, N),
        'reco': np.random.normal(50, 10, N),
        'true': np.random.normal(50, 10, N),
        'weight': np.random.rand(N)
    })
    
    # Example: Specify bins using np.linspace.
    custom_bins = np.linspace(100, 1000, 20)
    plot_difference(df_example, x_value_name='energy', 
                                y_value_reco='reco', y_value_true='true',
                                bins=custom_bins,
                                xscale='linear', yscale='linear', 
                                xlim=(100, 1000), ylim=(0, 30))
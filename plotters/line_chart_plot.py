import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_line_chart(x, y, y_err=None, marker="o", linestyle="-", color=None, label=None,
                    ax=None, xscale=None, yscale=None, xlim=None, ylim=None):
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
    ax : matplotlib.axes.Axes or None, optional
        Axis on which to plot. If None, a new figure and axis are created.
    xscale : str or None, optional
        Scale for the x-axis (e.g., 'linear' or 'log').
    yscale : str or None, optional
        Scale for the y-axis.
    xlim : tuple or None, optional
        Limits for the x-axis as (min, max).
    ylim : tuple or None, optional
        Limits for the y-axis as (min, max).

    Returns
    -------
    tuple
        If a new axis is created, returns (fig, ax); if an existing axis is used, returns (None, ax).
    """
    new_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        new_fig = True
    else:
        fig = None

    # Plot the line. Let matplotlib choose a default color if none is provided.
    if color is None:
        lines = ax.plot(x, y, marker=marker, linestyle=linestyle, label=label)
        actual_color = lines[0].get_color()
    else:
        ax.plot(x, y, marker=marker, linestyle=linestyle, color=color, label=label)
        actual_color = color

    # Plot error bands if y_err is provided.
    if y_err is not None:
        ax.fill_between(x, y - y_err, y + y_err, color=actual_color, alpha=0.3)

    # Set axis scales and limits if provided.
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if label:
        ax.legend()

    if new_fig:
        return fig, ax
    else:
        return None, ax
    
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
                             weight_name="weight", labels=None, colors=None,
                             bins=None, xscale=None, yscale=None, xlim=None, ylim=None,
                             show=True, save_path=None):
    """
    Plot weighted differences for multiple reconstructed y-value columns on one plot.

    For each column in y_value_reco, the function computes the absolute difference
    with the true value, bins the data along x_value_name, computes the weighted mean
    and standard deviation in each bin, and then plots the results with error bands.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    x_value_name : str
        Column name for the x-axis (used for binning).
    y_value_reco : list or str
        List of column names for the reconstructed y-values. If a single string is provided,
        it will be converted to a list.
    y_value_true : str
        Column name for the true y-values.
    weight_name : str, optional
        Column name for the weight values (default is "weight").
    labels : list or str, optional
        Labels for each dataset. If not provided, defaults to "Difference <col>".
    colors : list or str, optional
        Colors for each dataset. If a single string or None is provided, it is replicated.
    bins : array-like or None, optional
        Pre-specified bins for the x-axis. If None, defaults to np.logspace(2, 8, 20).
    xscale : str or None, optional
        Scale for the x-axis.
    yscale : str or None, optional
        Scale for the y-axis.
    xlim : tuple or None, optional
        x-axis limits as (min, max).
    ylim : tuple or None, optional
        y-axis limits as (min, max).
    show : bool, default True
        Whether to display the plot.
    save_path : str or None, optional
        If provided, the plot is saved to this path.

    Returns
    -------
    fig, ax : tuple
        The matplotlib figure and axes objects.
    """
    # Ensure y_value_reco is a list.
    if not isinstance(y_value_reco, list):
        y_value_reco = [y_value_reco]
    n = len(y_value_reco)

    # Process labels.
    if labels is None:
        labels = [f"Difference {col}" for col in y_value_reco]
    elif isinstance(labels, str):
        labels = [labels] * n
    elif len(labels) != n:
        raise ValueError("Length of labels must equal length of y_value_reco list.")

    # Process colors.
    if colors is None:
        colors = [None] * n
    elif isinstance(colors, str):
        colors = [colors] * n
    elif len(colors) != n:
        raise ValueError("Length of colors must equal length of y_value_reco list.")

    # Use provided bins or default to np.logspace(2, 8, 20).
    if bins is None:
        bins = np.logspace(2, 8, 20)
    x_centers = (bins[:-1] + bins[1:]) / 2

    # Create one figure and axis.
    fig, ax = plt.subplots()

    # Loop over each reconstructed y-value column.
    for i, col in enumerate(y_value_reco):
        # Compute the absolute difference for this column.
        diff = np.abs(df[col] - df[y_value_true])
        temp = pd.DataFrame({
            x_value_name: df[x_value_name],
            "difference": diff,
            "weight": df[weight_name]
        })
        # Bin the data.
        temp["x_bin"] = pd.cut(temp[x_value_name], bins=bins, labels=False)
        # Compute weighted statistics for each bin.
        results = temp.groupby("x_bin").apply(
            lambda group: weighted_stats(group["difference"], group["weight"])
        ).reindex(range(len(bins) - 1), fill_value=(np.nan, np.nan))
        weighted_means = results.map(lambda x: x[0])
        weighted_std_devs = results.map(lambda x: x[1])
        valid = ~np.isnan(weighted_means)
        # Plot on the existing axis.
        plot_line_chart(x_centers[valid], weighted_means[valid],
                        y_err=weighted_std_devs[valid],
                        marker="o", linestyle="-", color=colors[i], label=labels[i],
                        ax=ax, xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim)

    # Set overall axis labels and title.
    ax.set_xlabel(x_value_name)
    ax.set_ylabel("Difference")
    ax.set_title("Difference Plot")
    ax.legend()

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
        'MCTruth_Cascade_Distance_value': np.random.uniform(100, 1000, N),
        'Taupede_spice3_Distance_value': np.random.normal(50, 10, N),
        'weight': np.random.rand(N)
    })

    # For a single reconstructed column (even as a list)
    Taupede_name_list = ['Taupede_spice3']
    suffix = "_Distance_value"
    y_value_reco = [item + suffix for item in Taupede_name_list]
    y_value_true = 'MCTruth_Cascade_Distance_value'
    
    # Plot multiple differences on one plot.
    plot_difference(df_example, 
                             x_value_name='MCTruth_Cascade_Distance_value', 
                             y_value_reco=y_value_reco, 
                             y_value_true=y_value_true,
                             xscale='linear', yscale='linear',
                             xlim=(100, 1000), ylim=(0, 30))
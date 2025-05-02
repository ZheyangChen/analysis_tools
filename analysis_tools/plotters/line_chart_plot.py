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

def plot_difference(data_input,
                    x_value_name,
                    y_value_reco,
                    y_value_true,
                    weight_name="weight",
                    labels=None,
                    colors=None,
                    bins=None,
                    xscale=None,
                    yscale=None,
                    xlim=None,
                    ylim=None,
                    show=True,
                    save_path=None):
    """
    Plot weighted differences for one or more datasets and one or more reconstructed columns.

    Parameters
    ----------
    data_input : pd.DataFrame or list of DataFrames or dict of DataFrames
        If DataFrame: behaves as before.
        If list:  will be auto‐labeled "df_0", "df_1", …
        If dict:  keys are used as dataset labels.
    x_value_name : str
        Column name for the x-axis (used for binning).
    y_value_reco : str or list of str
        Column(s) for reconstructed y-values.
    y_value_true : str
        Column name for the true y-value.
    weight_name : str
    labels : list of str, optional
        One label per curve.  Defaults to "<dataset>: Difference <col>".
    colors : list of colors or single color, optional
    bins : array-like, optional
        Defaults to np.logspace(2, 8, 20).
    (…rest as before…)
    """
    # --- Normalize data_input to a dict of DataFrames ---
    if isinstance(data_input, pd.DataFrame):
        data_dict = {"": data_input}
    elif isinstance(data_input, list):
        data_dict = {f"df_{i}": df for i, df in enumerate(data_input)}
    elif isinstance(data_input, dict):
        data_dict = data_input
    else:
        raise ValueError("data_input must be DataFrame, list, or dict of DataFrames")

    # --- Normalize y_value_reco to a list ---
    if isinstance(y_value_reco, str):
        y_value_reco = [y_value_reco]
    n_vars = len(y_value_reco)
    n_data = len(data_dict)
    total_curves = n_data * n_vars

    # --- Default bins ---
    if bins is None:
        bins = np.logspace(2, 8, 20)
    x_centers = (bins[:-1] + bins[1:]) / 2

    # --- Build default labels if needed ---
    default_labels = []
    for dlabel in data_dict:
        for col in y_value_reco:
            prefix = f"{dlabel}: " if dlabel else ""
            default_labels.append(f"{prefix}Difference {col}")

    if labels is None:
        labels = default_labels
    elif isinstance(labels, str):
        labels = [labels] * total_curves
    elif len(labels) != total_curves:
        raise ValueError(f"labels must have length {total_curves}")

    # --- Build default colors list ---
    if colors is None:
        colors = [None] * total_curves
    elif not isinstance(colors, (list, tuple)):
        colors = [colors] * total_curves
    elif len(colors) != total_curves:
        raise ValueError(f"colors must have length {total_curves}")

    # --- Plotting setup ---
    fig, ax = plt.subplots()
    curve_idx = 0

    for dlabel, df in data_dict.items():
        for col in y_value_reco:
            # compute absolute difference
            diff = np.abs(df[col] - df[y_value_true])
            temp = pd.DataFrame({
                x_value_name: df[x_value_name],
                "difference": diff,
                "weight": df[weight_name]
            })
            # bin and compute weighted stats
            temp["x_bin"] = pd.cut(temp[x_value_name], bins=bins, labels=False)
            results = (
                temp
                .groupby("x_bin")
                .apply(lambda g: weighted_stats(g["difference"], g["weight"]))
                .reindex(range(len(bins)-1), fill_value=(np.nan, np.nan))
            )
            means = results.map(lambda t: t[0])
            stds  = results.map(lambda t: t[1])
            valid = ~np.isnan(means)

            # plot_line_chart handles errorbars and scales
            plot_line_chart(
                x_centers[valid],
                means[valid],
                y_err=stds[valid],
                marker="o",
                linestyle="-",
                color=colors[curve_idx],
                label=labels[curve_idx],
                ax=ax,
                xscale=xscale,
                yscale=yscale,
                xlim=xlim,
                ylim=ylim
            )
            curve_idx += 1

    # final formatting
    ax.set_xlabel(x_value_name)
    ax.set_ylabel("Difference")
    ax.set_title("Difference Plot")
    ax.legend(loc="best")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

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
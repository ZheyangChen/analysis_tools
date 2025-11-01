import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Union, Sequence, Callable, Dict, Any

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



def plot_difference(
    data_input: Union[pd.DataFrame, Sequence[pd.DataFrame], Dict[str, pd.DataFrame]],
    x_value_name: str,
    y_value_reco: Union[str, Sequence[str]],
    y_value_true: str,
    weight_name: str = "weight",
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    bins: Optional[Sequence[float]] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    x_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    y_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot weighted differences for one or more reconstructed y-value columns on one plot.

    Parameters
    ----------
    data_input : DataFrame or list/dict of DataFrames
    x_value_name : str
        Column to bin on the x-axis.
    y_value_reco : str or list of str
        Reconstructed variable(s).
    y_value_true : str
        True variable.
    weight_name : str
    labels : list of str, optional
    colors : list of str, optional
    bins : array-like, optional
    xscale, yscale : 'linear'|'log' or None
    xlim, ylim : tuple or None
    xlabel, ylabel, title : str or None
    x_transform : callable, optional
    y_transform : callable, optional
    show : bool
    save_path : str or None
    """
    # normalize data_input → dict
    if isinstance(data_input, pd.DataFrame):
        data_dict = {"": data_input}
    elif isinstance(data_input, list):
        data_dict = {f"df_{i}": df for i, df in enumerate(data_input)}
    elif isinstance(data_input, dict):
        data_dict = data_input
    else:
        raise ValueError("data_input must be DataFrame, list, or dict")

    # normalize y_value_reco → list
    if isinstance(y_value_reco, str):
        y_cols = [y_value_reco]
    else:
        y_cols = list(y_value_reco)
    n_curves = len(data_dict) * len(y_cols)

    # default bins
    if bins is None:
        bins = np.linspace(0, 1, 20)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # default labels/colors
    default_labels = []
    for dkey in data_dict:
        for yc in y_cols:
            prefix = f"{dkey}: " if dkey else ""
            default_labels.append(f"{prefix}{yc}")
    if labels is None:
        labels = default_labels
    if colors is None:
        colors = [None] * n_curves
    if len(labels) != n_curves or len(colors) != n_curves:
        raise ValueError("labels/colors must match number of curves")

    # prepare figure
    fig, ax = plt.subplots()
    idx = 0
    for dkey, df in data_dict.items():
        # transform x once per dataset
        x = df[x_value_name].values
        if x_transform:
            x = x_transform(x)

        for yc in y_cols:
            reco = df[yc].values
            true = df[y_value_true].values
            if y_transform:
                reco = y_transform(reco)
                true = y_transform(true)

            diff = np.abs(reco - true)
            tmp = pd.DataFrame({
                x_value_name: x,
                "diff": diff,
                "w": df[weight_name].values
            })
            tmp["bin"] = pd.cut(tmp[x_value_name], bins=bins, labels=False)

            stats = (
                tmp.groupby("bin")
                   .apply(lambda g: weighted_stats(g["diff"], g["w"]))
                   .reindex(range(len(bins)-1), fill_value=(np.nan, np.nan))
            )
            means = stats.map(lambda t: t[0])
            stds  = stats.map(lambda t: t[1])
            valid = ~np.isnan(means)

            plot_line_chart(
                centers[valid],
                means[valid],
                y_err=stds[valid],
                ax=ax,
                marker="o",
                linestyle="-",
                color=colors[idx],
                label=labels[idx],
                xscale=xscale,
                yscale=yscale,
                xlim=xlim,
                ylim=ylim
            )
            idx += 1

    # final decorations
    ax.set_xlabel(xlabel or x_value_name)
    ax.set_ylabel(ylabel or "Difference")
    ax.set_title(title or "")
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
        
        
def plot_binned_purity(df, variable,
                       is_signal="is_signal",
                       weight_col="weight",
                       bins=20,
                       xlabel=None, ylabel="Purity",
                       title="Binned Purity", xlim=None, ylim=(0,1),
                       xscale='linear', yscale='linear',
                       show=True):
    """
    Plot signal purity in bins of a given variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    variable : str
        Column name of the variable to bin.
    is_signal : str, Series, or callable
        Signal definition. Can be:
          - str: column name for signal flags (==1 is signal)
          - Series or boolean array: True for signal rows
          - function: a function(row) -> bool, applied to each row
    weight_col : str
        Column name for weights.
    bins : int or sequence
        Binning for the variable.
    xlabel, ylabel, title : str
        Axis labels and plot title.
    xlim, ylim : tuple
        Axis ranges.
    xscale, yscale : str
        Scale type (e.g., 'linear', 'log').
    """
    if is_signal is None:
        raise ValueError("You must specify `is_signal` as a column name, mask, or function.")
    elif isinstance(is_signal, str):
        signal_mask = df[is_signal] == 1
    elif callable(is_signal):
        signal_mask = df.apply(is_signal, axis=1)
    elif isinstance(is_signal, (pd.Series, np.ndarray)):
        signal_mask = is_signal
    else:
        raise TypeError("`is_signal` must be a str, function, or boolean mask.")

    var = df[variable]
    w   = df[weight_col]

    # Total and signal weights per bin
    total, edges = np.histogram(var, bins=bins, weights=w)
    signal, _    = np.histogram(var[signal_mask], bins=edges, weights=w[signal_mask])

    # Purity
    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.divide(signal, total, out=np.zeros_like(signal, dtype=float), where=total>0)

    centers = 0.5 * (edges[:-1] + edges[1:])

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(centers, purity, marker="o", linestyle="-", label="Purity")
    plt.xlabel(xlabel or variable)
    plt.ylabel(ylabel)
    plt.title(title)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    if xscale: plt.xscale(xscale)
    if yscale: plt.yscale(yscale)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()

    return centers, purity
        
        
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
    
    
    # Example plot for purity plot
    # 1. With a column name
    plot_binned_purity(df_example, "score", is_signal="true_label")

    # 2. With a mask
    plot_binned_purity(df_example, "energy", is_signal=(df["label"] == "signal"))

    # 3. With a function
    plot_binned_purity(df_example, "zenith", is_signal=lambda row: row["pid"] == 15 and row["energy"] > 1e5)
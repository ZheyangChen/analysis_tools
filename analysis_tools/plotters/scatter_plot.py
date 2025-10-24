import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

def scatter_plot(
    plot_df: pd.DataFrame,
    x: str,
    y: str,
    bins=None, xbins=None, ybins=None,         # separate binning supported
    cuts=None, weight: str = 'weight',
    xscale: str = None, yscale: str = None,
    xlim=None, ylim=None,                       # axis limits
    cmap: str = 'viridis', color_scale: str = 'log',   # 'log' or 'linear'
    show_line: bool = False, line_color: str = 'r', line_label: str = None,
    xlabel: str = None, ylabel: str = None, title: str = None,
    colorbar_label: str = None, legend_loc: str = 'best',
    min_weight_threshold: float = 0.0,
    ax: plt.Axes = None, show: bool = True
) -> tuple[plt.Figure, plt.Axes]:

    # --- Build a position-aligned boolean mask (no reindexing) ---
    if cuts is None:
        mask = np.ones(len(plot_df), dtype=bool)
    elif callable(cuts):
        m = cuts(plot_df)
        mask = np.asarray(m.values if isinstance(m, pd.Series) else m, dtype=bool)
    else:
        # accept array/list/Series; ignore index and align by position
        arr = cuts.values if isinstance(cuts, pd.Series) else cuts
        mask = np.asarray(arr, dtype=bool)

    if mask.ndim != 1 or mask.shape[0] != len(plot_df):
        raise ValueError(f"`cuts` must be 1-D of length {len(plot_df)}.")

    # --- Pick norm for color scale ---
    if color_scale == 'log':
        norm = LogNorm()
    elif color_scale == 'linear':
        norm = Normalize()
    else:
        raise ValueError("color_scale must be 'log' or 'linear'")

    # --- Decide binning safely (no boolean ops on arrays) ---
    if xbins is not None or ybins is not None:
        if xbins is None or ybins is None:
            raise ValueError("If using separate binning, provide both xbins and ybins.")
        bins_x, bins_y = xbins, ybins
    else:
        if bins is None:
            bins_x = bins_y = 50
        elif isinstance(bins, (list, tuple)) and len(bins) == 2:
            bins_x, bins_y = bins
        else:
            bins_x = bins_y = bins  # scalar int or 1D array

    # --- Pull columns as NumPy arrays (position-aligned) ---
    xvals = np.asarray(plot_df[x])[mask]
    yvals = np.asarray(plot_df[y])[mask]
    wvals = np.asarray(plot_df[weight])[mask]

    # --- Compute 2D histogram ---
    counts, xedges, yedges = np.histogram2d(
        xvals, yvals,
        bins=[bins_x, bins_y],
        weights=wvals
    )

    # --- Threshold tiny-weight bins (avoid log(0)) ---
    counts = np.where(counts >= min_weight_threshold, counts, np.nan)

    # --- Figure / axes ---
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # --- Plot ---
    img = ax.pcolormesh(xedges, yedges, counts.T, norm=norm, cmap=cmap)

    # --- Optional y = x line ---
    if show_line:
        # Use x-limits if provided, else the histogram grid extents
        xmin = xlim[0] if xlim is not None else xedges[0]
        xmax = xlim[1] if xlim is not None else xedges[-1]
        ax.plot([xmin, xmax], [xmin, xmax],
                color=line_color, label=line_label or 'y = x')

    # --- Scales and limits ---
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if xlim:   ax.set_xlim(xlim)
    if ylim:   ax.set_ylim(ylim)

    # --- Labels & title ---
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title: ax.set_title(title)
    if show_line and line_label:
        ax.legend(loc=legend_loc)

    # --- Colorbar ---
    cbar = fig.colorbar(img, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

# Example usage:
if __name__ == '__main__':
    # Create dummy data
    np.random.seed(0)
    N = 1000
    df = pd.DataFrame({
        'energy_x': np.random.normal(50, 10, N),
        'energy_y': np.random.normal(50, 15, N),
        'weight': np.random.rand(N)
    })
    
    # Call scatter_energy with various options
    scatter_energy(df, 'energy_x', 'energy_y', bins=50,
                   cuts=(df['energy_x'] > 30) & (df['energy_x'] < 70),
                   xscale='linear', yscale='linear',
                   cmap='plasma', show_line=True, line_color='blue',
                   line_label='y=x', xlabel='Energy X [units]',
                   ylabel='Energy Y [units]', title='Scatter Plot with 2D Histogram',
                   colorbar_label='Counts', legend_loc='upper left')
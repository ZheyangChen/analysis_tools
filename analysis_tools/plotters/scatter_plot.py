from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm, Normalize

def scatter_plot(
    plot_df, x, y, bins, cuts=None, weight='weight',
    xscale=None, yscale=None, cmap='viridis',
    color_scale: str = 'log',      # 'log' or 'linear'
    show_line:   bool  = False,    # plot y=x inside?
    line_color:  str   = 'r',
    line_label:  str   = None,
    xlabel:      str   = None,
    ylabel:      str   = None,
    title:       str   = None,
    colorbar_label: str = None,
    legend_loc:  str   = 'best',
    min_weight_threshold: float = 0.0,
    ax:          plt.Axes = None,  # <-- NEW
    show:        bool      = True  # <-- NEW
) -> (plt.Figure, plt.Axes):
    """
    Same as before, but:
      • if ax is provided, plot into it; else make a new one
      • if show=False, don't call plt.show()
    """
    # 1) Mask
    mask = cuts if cuts is not None else np.ones(len(plot_df), dtype=bool)

    # 2) Choose norm
    if   color_scale == 'log':    norm = LogNorm()
    elif color_scale == 'linear': norm = Normalize()
    else: raise ValueError("color_scale must be 'log' or 'linear'")

    # 3) Compute 2D histogram
    counts, xedges, yedges = np.histogram2d(
        plot_df.loc[mask, x],
        plot_df.loc[mask, y],
        bins=bins,
        weights=plot_df.loc[mask, weight]
    )
    # 4) Mask out low-weight bins
    counts = np.where(counts >= min_weight_threshold, counts, np.nan)

    # 5) Figure / Axes setup
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # 6) Plot via pcolormesh
    img = ax.pcolormesh(
        xedges, yedges, counts.T,
        norm=norm, cmap=cmap
    )

    # 7) Optional reference line inside function
    if show_line:
        if hasattr(bins, '__len__'):
            rmin, rmax = bins[0], bins[-1]
        else:
            rmin, rmax = plot_df[x].min(), plot_df[x].max()
        ax.plot([rmin, rmax], [rmin, rmax],
                color=line_color, label=line_label or 'y = x')

    # 8) Axes decor
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:   ax.set_title(title)
    if show_line and line_label:
        ax.legend(loc=legend_loc)

    # 9) Colorbar
    cbar = fig.colorbar(img, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    # 10) Finalize
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
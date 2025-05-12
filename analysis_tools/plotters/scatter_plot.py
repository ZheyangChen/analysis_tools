import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


def scatter_plot(plot_df, x, y, bins, cuts=None, weight='weight',
                 xscale=None, yscale=None, cmap='viridis',
                 color_scale: str = 'log',      # 'log' or 'linear'
                 show_line=False, line_color='r', line_label=None,
                 xlabel=None, ylabel=None, title=None,
                 colorbar_label=None, legend_loc='best'):
    """
    Create a 2D histogram (scatter-like plot) with optional reference line
    and either a logarithmic or linear colorbar.

    Returns (fig, ax) so you can further customize if needed.
    """
    # 1) Mask
    mask = cuts if cuts is not None else np.ones(len(plot_df), dtype=bool)

    # 2) Choose norm
    if color_scale == 'log':
        norm = LogNorm()
    elif color_scale == 'linear':
        norm = Normalize()
    else:
        raise ValueError("color_scale must be 'log' or 'linear'")

    # 3) Draw
    fig, ax = plt.subplots()
    counts, xedges, yedges, image = ax.hist2d(
        plot_df.loc[mask, x],
        plot_df.loc[mask, y],
        bins=bins,
        weights=plot_df.loc[mask, weight],
        norm=norm,
        cmap=cmap
    )

    # 4) Reference line
    if show_line:
        if hasattr(bins, '__len__'):
            rmin, rmax = bins[0], bins[-1]
        else:
            rmin, rmax = plot_df[x].min(), plot_df[x].max()
        ax.plot([rmin, rmax], [rmin, rmax],
                color=line_color,
                label=line_label or 'Reference')

    # 5) Axes scales, labels, title
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if show_line and line_label:
        ax.legend(loc=legend_loc)

    # 6) Colorbar
    cbar = fig.colorbar(image, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    fig.tight_layout()
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd

def scatter_plot(plot_df, x, y, bins, cuts=None, weight='weight',
                   xscale=None, yscale=None, cmap='viridis',
                   show_line=False, line_color='r', line_label=None,
                   xlabel=None, ylabel=None, title=None, colorbar_label=None,
                   legend_loc='best'):
    """
    Create a 2D histogram (scatter-like plot) with an optional reference line.
    
    Parameters
    ----------
    plot_df : pandas.DataFrame
        DataFrame containing the data.
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.
    bins : int or sequence
        Number of bins or array of bin edges to use in both dimensions.
    cuts : array-like of bool, optional
        Boolean mask to filter the DataFrame. If None, no filtering is applied.
    weight : str, default 'weight'
        Column name for weights.
    xscale : str or None
        Scale for the x-axis (e.g., 'linear' or 'log').
    yscale : str or None
        Scale for the y-axis.
    cmap : str, default 'viridis'
        Colormap for the 2D histogram.
    show_line : bool, default False
        If True, plot a reference line (default is y=x).
    line_color : str, default 'r'
        Color for the reference line.
    line_label : str or None
        Label for the reference line in the legend.
    xlabel : str or None
        X-axis label; defaults to the x column name.
    ylabel : str or None
        Y-axis label; defaults to the y column name.
    title : str or None
        Plot title.
    colorbar_label : str or None
        Label for the colorbar.
    legend_loc : str, default 'best'
        Location for the legend.
    
    Returns
    -------
    None
    """
    # If no cuts provided, create a mask that selects all rows.
    if cuts is not None:
        mask = cuts
    else:
        mask = np.ones(len(plot_df), dtype=bool)
    
    # Create figure and axis.
    fig, ax = plt.subplots()
    
    # Create the 2D histogram.
    h = ax.hist2d(plot_df.loc[mask, x], plot_df.loc[mask, y],
                  bins=bins, weights=plot_df.loc[mask, weight],
                  norm=LogNorm(), cmap=cmap)
    
    # Optionally plot a reference line (e.g., y=x).
    if show_line:
        # Here we assume you want a line over the bin range.
        # If bins is a sequence, use its first and last values; otherwise, derive from data.
        if hasattr(bins, '__len__'):
            ref_min, ref_max = bins[0], bins[-1]
        else:
            ref_min, ref_max = plot_df[x].min(), plot_df[x].max()
        ax.plot([ref_min, ref_max], [ref_min, ref_max], color=line_color,
                label=line_label if line_label else 'Reference')
    
    # Set axis scales if provided.
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    
    # Set axis labels.
    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)
    
    # Set title if provided.
    if title:
        ax.set_title(title)
    
    # Add legend if a reference line is drawn.
    if show_line and line_label:
        ax.legend(loc=legend_loc)
    
    # Add colorbar and set its label if provided.
    cbar = fig.colorbar(h[3], ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    
    plt.tight_layout()
    plt.show()

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
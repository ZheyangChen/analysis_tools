import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_line(plot_df, x, y, sort_data=False,
              xlabel=None, ylabel=None, title=None, legend_loc='best',
              line_style='-', color=None, label=None,
              show=True, save_path=None):
    """
    Plot a line plot from a pandas DataFrame.
    
    If sort_data is True, the DataFrame is sorted by the x-axis variable before plotting.
    
    Parameters
    ----------
    plot_df : pandas.DataFrame
        DataFrame containing the data.
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.
    sort_data : bool, default False
        If True, sort the DataFrame by the x-axis variable.
    xlabel : str or None
        Label for the x-axis; defaults to the x column name if None.
    ylabel : str or None
        Label for the y-axis; defaults to the y column name if None.
    title : str or None
        Plot title.
    legend_loc : str, default 'best'
        Location for the legend.
    line_style : str, default '-'
        Style of the line (e.g., '-', '--', ':').
    color : str or None
        Color of the line.
    label : str or None
        Label for the line (used in the legend).
    show : bool, default True
        Whether to display the plot.
    save_path : str or None
        If provided, save the plot to this path.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """
    # Sort the data if required.
    df = plot_df.sort_values(by=x) if sort_data else plot_df

    # Create figure and axis.
    fig, ax = plt.subplots()
    
    # Plot the line.
    ax.plot(df[x], df[y], linestyle=line_style, color=color, label=label)
    
    # Set axis labels.
    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)
    
    # Set title if provided.
    if title:
        ax.set_title(title)
    
    # Add legend if label is provided.
    if label:
        ax.legend(loc=legend_loc)
    
    plt.tight_layout()
    
    # Save or show the plot.
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_bar(plot_df, x, y, xlabel=None, ylabel=None, title=None,
             legend_loc='best', bar_width=0.8, color=None, label=None,
             show=True, save_path=None):
    """
    Plot a bar plot from a pandas DataFrame.
    
    Parameters
    ----------
    plot_df : pandas.DataFrame
        DataFrame containing the data.
    x : str
        Column name for the x-axis (categorical or numerical).
    y : str
        Column name for the y-axis.
    xlabel : str or None
        Label for the x-axis; defaults to the x column name if None.
    ylabel : str or None
        Label for the y-axis; defaults to the y column name if None.
    title : str or None
        Plot title.
    legend_loc : str, default 'best'
        Location for the legend.
    bar_width : float, default 0.8
        Width of the bars.
    color : str or None
        Color of the bars.
    label : str or None
        Label for the data series (used in the legend).
    show : bool, default True
        Whether to display the plot.
    save_path : str or None
        If provided, save the plot to this path.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """
    # Create figure and axis.
    fig, ax = plt.subplots()
    
    # Plot the bars.
    ax.bar(plot_df[x], plot_df[y], width=bar_width, color=color, label=label)
    
    # Set axis labels.
    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)
    
    # Set title if provided.
    if title:
        ax.set_title(title)
    
    # Add legend if label is provided.
    if label:
        ax.legend(loc=legend_loc)
    
    plt.tight_layout()
    
    # Save or show the plot.
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

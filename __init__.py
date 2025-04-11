# analysis_tools/__init__.py

"""
analysis_tools package

This package provides various tools for analysis including plotting functions
and data selection utilities.
"""

# Import functions from the plotters package.
from .plotters.histogram_plot import plot_histograms,plot_stacked_hist_with_ratio
from .plotters.line_chart_plot import plot_difference
from .plotters.scatter_plot import scatter_plot

from .plotters.multi_plots import plot_multi_var

# Import functions from the my_selectors package.
#from .my_selectors.select_data import select_data_with_operators
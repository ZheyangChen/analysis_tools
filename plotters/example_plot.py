# Example usage histogram plot:
if __name__ == '__main__':
    import pandas as pd
    # Create some dummy data for demonstration
    bins = np.linspace(-2000, 200, 50)
    plotvar = 'cscdSBU_MonopodFit4_noDC_Delay_ice_value'
    
    # Dummy data frames (replace with your actual data)
    nutau_all = pd.DataFrame({plotvar: np.random.normal(-800, 300, 1000),
                              'astro_weight': np.random.rand(1000)})
    nue_all = pd.DataFrame({plotvar: np.random.normal(-600, 250, 1000),
                            'astro_weight': np.random.rand(1000),
                            'conv_weight': np.random.rand(1000)})
    numu_all = pd.DataFrame({plotvar: np.random.normal(-400, 200, 1000),
                             'astro_weight': np.random.rand(1000),
                             'conv_weight': np.random.rand(1000)})
    
    data_dict = {
        'nutau': nutau_all,
        'nue': nue_all,
        'numu': numu_all
    }
    weights_map = {
        'nutau': 'astro_weight',
        'nue': ['astro_weight', 'conv_weight'],
        'numu': ['astro_weight', 'conv_weight']
    }
    
    plot_histograms(data_dict, plotvar, bins, weights_map)

    
    
# Example hist with ratio:
if __name__ == '__main__':
    # Create dummy data
    import pandas as pd
    bins = np.linspace(0, 100, 21)
    plotvar = 'measurement'
    
    # Dummy DataFrames with a weight column
    df1 = pd.DataFrame({
        plotvar: np.random.normal(50, 10, 1000),
        'weight': np.random.rand(1000)
    })
    df2 = pd.DataFrame({
        plotvar: np.random.normal(60, 15, 1000),
        'weight': np.random.rand(1000)
    })
    data_dict = {'Dataset 1': df1, 'Dataset 2': df2}
    weights_map = {'Dataset 1': 'weight', 'Dataset 2': 'weight'}
    
    # Plot non-normalized histogram
    plot_histograms(data_dict, plotvar, bins, weights_map, normalized=False,
                    xlabel='Measurement', ylabel='Rate per Year', title='Raw Histogram')
    
    # Plot normalized histogram
    plot_histograms(data_dict, plotvar, bins, weights_map, normalized=True,
                    xlabel='Measurement', ylabel='Probability Density', title='Normalized Histogram')
    
    
    
# Example scattering plot:
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
    scatter_plot(df, 'energy_x', 'energy_y', bins=50,
                   cuts=(df['energy_x'] > 30) & (df['energy_x'] < 70),
                   xscale='linear', yscale='linear',
                   cmap='plasma', show_line=True, line_color='blue',
                   line_label='y=x', xlabel='Energy X [units]',
                   ylabel='Energy Y [units]', title='Scatter Plot with 2D Histogram',
                   colorbar_label='Counts', legend_loc='upper left')
    
    
    
# Example usage:
if __name__ == '__main__':
    # Create example data for a line plot.
    np.random.seed(0)
    N = 50
    df_line = pd.DataFrame({
        'time': np.linspace(0, 10, N),
        'measurement': np.random.rand(N) * 100
    })
    
    # Create example data for a bar plot.
    df_bar = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D'],
        'value': [10, 15, 7, 20]
    })
    
    # Line plot with sorting enabled.
    plot_line(df_line, 'time', 'measurement', sort_data=True,
              xlabel='Time [s]', ylabel='Measurement', title='Line Plot Example',
              color='green', label='Measurement')
    
    # Bar plot example.
    plot_bar(df_bar, 'category', 'value',
             xlabel='Category', ylabel='Value', title='Bar Plot Example',
             color='skyblue', label='Value')
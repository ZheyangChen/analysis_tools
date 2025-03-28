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

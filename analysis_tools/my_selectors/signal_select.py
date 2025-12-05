from analysis_tools.my_selectors.apply_selection import apply_selection, make_mask_from_criteria
from analysis_tools.workflows.prediction_flow import prediction_flow


def add_postBDT_variables(df):
    df['BDT_score_product'] = df['bdt1_score']*df['bdt2_score']
    df['Mono_E_charge_ratio'] = df['cscdSBU_MonopodFit4_noDC_energy']/df['cscdSBU_Qtot_HLC_value']
    df['Tau1_E_charge_ratio'] = df['Taupede_ftp_1Particles_energy']/df['cscdSBU_Qtot_HLC_value']
    df['Tau2_E_charge_ratio'] = df['Taupede_ftp_2Particles_energy']/df['cscdSBU_Qtot_HLC_value']
    return df

def signal_select(df, BDT_model, BDT_features, BDT_threholds, cut_dict):
    df = prediction_flow(
        df,
        BDT_model,
        BDT_features,
        thresholds=BDT_threholds
        )
    df = add_postBDT_variables(df)
    #print(df.columns.tolist())
    
    df_BDTcut = df[(df['bdt1_score_pass']==1)&(df['bdt2_score_pass']==1)]
    final_df = apply_selection(df_BDTcut,cut_dict)
    return final_df


def BDT_feature_name_change(feature_list):
    #change some names used in old and new datasets
    replacements = {
    'TauMonoDiff_rlogl': 'TauMonoDiff_rlogl_value',
    'Taupede_Asymmetry': 'Taupede_Asymmetry_value',
    'Taupede_Distance': 'Taupede_Distance_value',
    'CascadeLlhVertexFitParams_rlogL':'cscdSBU_LE_bdt_input_CascadeLlhVertexFitParams_rlogL', 
    'CscdL3_SPEFit16FitParams_rlogl':'cscdSBU_LE_bdt_input_CscdL3_SPEFit16FitParams_rlogl', 
    'CscdL3_SPEFit16_zenith':'cscdSBU_LE_bdt_input_CscdL3_SPEFit16_zenith',
    'LineFit_zenith':'cscdSBU_LE_bdt_input_LineFit_zenith',
    'cscdSBU_I3XYScale_noDC_value':'cscdSBU_LE_bdt_input_cscdSBU_I3XYScale_noDC_value',
    'cscdSBU_L4StartingTrackHLC_cscdSBU_MonopodFit4_noDCVetoCharge_value':'cscdSBU_LE_bdt_input_cscdSBU_L4StartingTrackHLC_cscdSBU_MonopodFit4_noDCVetoCharge_value',
    'cscdSBU_L4VetoTrack_cscdSBU_MonopodFit4_noDCVetoCharge_value':'cscdSBU_LE_bdt_input_cscdSBU_L4VetoTrack_cscdSBU_MonopodFit4_noDCVetoCharge_value',
    'cscdSBU_MonopodFit4_noDC_Delay_ice_value':'cscdSBU_LE_bdt_input_cscdSBU_MonopodFit4_noDC_Delay_ice_value',
    'cscdSBU_MonopodFit4_noDC_z':'cscdSBU_LE_bdt_input_cscdSBU_MonopodFit4_noDC_z',
    'cscdSBU_MonopodFit4_noDC_zenith':'cscdSBU_LE_bdt_input_cscdSBU_MonopodFit4_noDC_zenith',
    'cscdSBU_Qtot_HLC_log_value':'cscdSBU_LE_bdt_input_cscdSBU_Qtot_HLC_log_value',
    'cscdSBU_VertexRecoDist_CscdLLh':'cscdSBU_LE_bdt_input_cscdSBU_VertexRecoDist_CscdLLh',
    'cscdSBU_VetoDepthFirstHit_value':'cscdSBU_LE_bdt_input_cscdSBU_VetoDepthFirstHit_value'
    }

    new_feature_list = [replacements.get(x, x) for x in feature_list]
    
    return new_feature_list
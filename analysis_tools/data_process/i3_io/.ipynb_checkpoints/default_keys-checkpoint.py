# analysis_tools/data_process/i3_io/default_keys.py

# Base keys
BASE_KEYS = [
    'I3EventHeader',
    'CVMultiplicity',
    'CVStatistics',
]


MC_KEYS = [
    'I3MCWeightDict',
    'cscdSBU_MCTruth',
    'cscdSBU_MCPrimary',
    'cscdSBU_AtmWeight_Conv',
    'cscdSBU_AtmWeight_Conv_PassRate',
    'cscdSBU_AtmWeight_Prompt',
    'cscdSBU_AtmWeight_Prompt_berss',
    'cscdSBU_AtmWeight_Prompt_PassRate',
]


CASCADE_KEYS = [
    'cscdSBU_MonopodFit4_noDC',
    'cscdSBU_MonopodFit4_noDCFitParams',
    'cscdSBU_MonopodFit4_noDC_Delay_ice',
    'cscdSBU_MaxQtotRatio_SplitInIcePulses',
    'cscdSBU_Qtot_HLC',
    'cscdSBU_L4StartingTrackHLC_cscdSBU_MonopodFit4_OfflinePulsesHLC_noDCVetoCharge',
    'cscdSBU_LE_bdt_cascade',
    'cscdSBU_LE_bdt_hybrid',
    'cscdSBU_LE_bdt_track',
    'cscdSBU_LE_bdt_input',
    'CscdL3_CascadeLlhVertexFit',
    'cscdSBU_MaxQtotRatio_HLC',
    'cscdSBU_VetoDepthFirstHit',
    'cscdSBU_VertexRecoDist_CscdLLh',
    'cscdSBU_VetoMaxDomChargeOM',
]


NEW_MONO_KEYS = [
    'MonopodFit_iMIGRAD_PPB0',
    'MonopodFit_iMIGRAD_PPB0FitParams',
    'MonopodFit_iMIGRAD_PPB0_Seed',
    'MonopodFit_iMIGRAD_PPB0AmpSeed',
    'MonopodFit_iMIGRAD_PPB0AmpSeedFitParams',
    'MonopodFit_iMIGRAD_PPB0_VertexRecoDist_CscdLLh',
]

TRUE_TAU_KEYS = [
    'Cascade1_vis_truth_tau',
    'Cascade2_vis_truth_tau',
    'MCTruth_Cascade_Distance',
    'MCTruth_Tau_Asymmetry',
    'TrueTau',
    'NuTaudecaytype',
    'cc',
]

TAUPEDE_KEYS = [
    'Taupede_spice3',
    'Taupede_spice3FitParams',
    'Taupede_spice3Particles',
    'Taupede1_spice3Particles',
    'Taupede2_spice3Particles',
    'Taupede_Distance',
    'Taupede_Asymmetry',
    'TauMonoDiff_rlogl',
    'Taupede_ftp',
    'Taupede_ftpFitParams',
    'Taupede_ftpParticles',
    'Taupede_ftp_1Particles ',
    'Taupede_ftp_2Particles ',
    'Taupede_ftp_Distance',
    'Taupede_ftp_Asymmetry',
    'Taupede_ftpMonoDiff_rlogl',
]

SNOWSTORM_KEYS = [
    'SnowstormParameterDict',
    'PolyplopiaPrimary',
    'penetrating_depth_v1_gcd',
]

# any additional keys you had inline
ADDITIONAL_KEYS = [
    'TotalWeight_glashowcorrection',
    'Energy_tau_lepton',
    'Energy_tau_decayproduct',
    'Number_tau_decaypions',
    'Number_tau_decaypi0',
    'Energy_tau_decaynutau',
]

# combine once here
DEFAULT_I3_KEYS = (
    BASE_KEYS
  + MC_KEYS
  + CASCADE_KEYS
  + TRUE_TAU_KEYS
  + TAUPEDE_KEYS
  + SNOWSTORM_KEYS
  + ADDITIONAL_KEYS
)

# if you want the “newreco_keys” & “Taupede_name_add_list” items,
# you can build them the same way and extend DEFAULT_I3_KEYS here.
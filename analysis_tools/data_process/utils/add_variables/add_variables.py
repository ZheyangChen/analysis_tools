import numpy as np
import pandas as pd

def distance(x1, x2, y1, y2, z1, z2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

def add_variables(df):
    if ('Taupede_Asymmetry_value' not in df.columns) & ('Taupede1_spice3Particles_x' in df.columns):
        print('Calculating Taupede variables based on spice3.2.1 ice model')
        tx1 = df['Taupede1_spice3Particles_x']
        ty1 = df['Taupede1_spice3Particles_y']
        tz1 = df['Taupede1_spice3Particles_z']
        tx2 = df['Taupede2_spice3Particles_x']
        ty2 = df['Taupede2_spice3Particles_y']
        tz2 = df['Taupede2_spice3Particles_z']
        df['Taupede_Distance_value'] = distance(tx1, tx2, ty1, ty2, tz1, tz2)
        E1 = df['Taupede1_spice3Particles_energy']
        E2 = df['Taupede2_spice3Particles_energy']
        df['Taupede_Asymmetry_value'] = (E1-E2)/(E1+E2)
        df['TauMonoDiff_rlogl_value'] = df['Taupede_spice3FitParams_rlogl'] - df['cscdSBU_MonopodFit4_noDCFitParams_rlogl']
    else:
        print('Taupede asymmetry value already calculated')
        

    if 'Cascade1_vis_truth_tau_energy' in df.columns:
        true_E1 = df['Cascade1_vis_truth_tau_energy']
        true_E2 = df['Cascade2_vis_truth_tau_energy']
        df['MCTruth_Tau_Asymmetry_value'] = (true_E1-true_E2)/(true_E1+true_E2)
        true_x1 = df['Cascade1_vis_truth_tau_x']
        true_y1 = df['Cascade1_vis_truth_tau_y']
        true_z1 = df['Cascade1_vis_truth_tau_z']
        true_x2 = df['Cascade2_vis_truth_tau_x']
        true_y2 = df['Cascade2_vis_truth_tau_y']
        true_z2 = df['Cascade2_vis_truth_tau_z']
        df['MCTruth_Cascade_Distance_value'] = distance(true_x1, true_x2, true_y1, true_y2, true_z1, true_z2)

    else:
        print('No true cascades, not a double cascade')

    if not 'cscdSBU_Qtot_HLC_log_value' in df.columns:
        df['cscdSBU_Qtot_HLC_log_value']=np.log10(df['cscdSBU_Qtot_HLC_value'])

    if ('Taupede_spice3' in df.columns) & ('Taupede_spice3_Asymmetry_value' not in df.columns):
        new_prefix_list = ['Taupede_spice3_1','Taupede_spice3_2']
        old_prefix_list = ['Taupede1_spice3','Taupede2_spice3']
        mapping = {}
        for old, new in zip(old_prefix_list, new_prefix_list):
            for col in df.columns:
                if col.startswith(old):
                    mapping[col] = new + col[len(old):] 
        mapping.update({
            "Taupede_Asymmetry_value": "Taupede_spice3_Asymmetry_value",
            "Taupede_Distance_value": "Taupede_spice3_Distance_value",
        })
        df.rename(columns=mapping, inplace=inplace) 

    if not 'cscdSBU_VertexRecoDist_CscdLLh' in df.columns:
        x1 = df['CscdL3_CascadeLlhVertexFit_x']
        y1 = df['CscdL3_CascadeLlhVertexFit_y']
        z1 = df['CscdL3_CascadeLlhVertexFit_z']
        x2 = df['cscdSBU_MonopodFit4_noDC_x']
        y2 = df['cscdSBU_MonopodFit4_noDC_y']
        z2 = df['cscdSBU_MonopodFit4_noDC_z']
        df['cscdSBU_VertexRecoDist_CscdLLh'] = distance(x1, x2, y1, y2, z1, z2)

    if 'cscdSBU_LE_bdt_input_cscdSBU_I3XYScale_noDC_value' not in df.columns:
        cascade_BDT_input_list = ['cscdSBU_LE_bdt_input_CascadeLlhVertexFitParams_rlogL', 
                              'cscdSBU_LE_bdt_input_CscdL3_SPEFit16FitParams_rlogl', 
                              'cscdSBU_LE_bdt_input_CscdL3_SPEFit16_zenith',
                              'cscdSBU_LE_bdt_input_LineFit_zenith',
                              'cscdSBU_LE_bdt_input_cscdSBU_I3XYScale_noDC_value',
                              'cscdSBU_LE_bdt_input_cscdSBU_L4StartingTrackHLC_cscdSBU_MonopodFit4_noDCVetoCharge_value',
                              'cscdSBU_LE_bdt_input_cscdSBU_L4VetoTrack_cscdSBU_MonopodFit4_noDCVetoCharge_value',
                              'cscdSBU_LE_bdt_input_cscdSBU_MonopodFit4_noDC_Delay_ice_value',
                              'cscdSBU_LE_bdt_input_cscdSBU_MonopodFit4_noDC_z',
                              'cscdSBU_LE_bdt_input_cscdSBU_MonopodFit4_noDC_zenith',
                              'cscdSBU_LE_bdt_input_cscdSBU_Qtot_HLC_log_value',
                              'cscdSBU_LE_bdt_input_cscdSBU_VertexRecoDist_CscdLLh',
                              'cscdSBU_LE_bdt_input_cscdSBU_VetoDepthFirstHit_value']
        for variable in cascade_BDT_input_list:
            df[variable] = df[variable[21:]]
    return df

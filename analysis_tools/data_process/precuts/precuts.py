import numpy as np
import pandas as pd

def precuts(infile):
    BDT_input_name = 'cscdSBU_LE_bdt_input_'
    cut1 = np.log10(infile['cscdSBU_MonopodFit4_noDC_energy']) > 4.5
    cut2 = np.log10(infile['cscdSBU_Qtot_HLC_value']) > 3
    cut3 = (infile['Taupede_Distance_value']>10)&(infile['Taupede_Distance_value']<400)
    cut4 = infile[BDT_input_name+'cscdSBU_I3XYScale_noDC_value']<0.9
    #cut5 = (infile['TauMonoDiff_rlogl']>-20)&(infile['TauMonoDiff_rlogl']<0)
    cut6 = (infile['cscdSBU_MonopodFit4_noDC_z']>-500)&(infile['cscdSBU_MonopodFit4_noDC_z']<500)
    cut7 = (infile['cscdSBU_MonopodFit4_noDC_z']>-50)|(infile['cscdSBU_MonopodFit4_noDC_z']<-150)
    cut = cut1 & cut2 &cut3&cut4&cut6&cut7
    outfile = infile[cut].copy()
    return outfile.reset_index(drop=True)




def singlekey_cut(df, column, conditions):

    #conditions : [(">", 60), ("<=", 80)]
    
    mask = pd.Series(True, index=df.index)
    
    for operator, value in conditions:
        if operator == ">":
            mask &= (df[column] > value)
        elif operator == ">=":
            mask &= (df[column] >= value)
        elif operator == "<":
            mask &= (df[column] < value)
        elif operator == "<=":
            mask &= (df[column] <= value)
        elif operator == "==":
            mask &= (df[column] == value)
        elif operator == "!=":
            mask &= (df[column] != value)
        else:
            raise ValueError(f"does not support: {operator}")
    out_df = df[mask].copy()
    
    return out_df.reset_index(drop=True)


import numpy as np
import glob

def add_nugen_weights(df, ltime,indir, sv="step340",astro_norm = 1.83,astro_index = -2.58, cosmicray="HillasGaisser2012_H4a", hadronicinteraction="SIBYLL2.3c"):
    print(cosmicray, hadronicinteraction)
    print(indir)
    nfiles = len(glob.glob(indir+f"/00*/final_cascade/*.i3.*"))
    if nfiles == 0:
        nfiles = len(glob.glob(indir+f"/p0=0.0_p1=0.0_domeff=1.00/00*/final_cascade/*.i3.*"))
    if nfiles == 0:
        nfiles = len(glob.glob(indir+f"/*.i3.*"))
    #if nfiles == 0:
    #    nfiles = len(glob.glob(indir + f"*/*.i3.*"))
    print('number of files ',nfiles)

    df['nfiles'] = np.ones(len(df.index))*nfiles
    
    df['astro_weight'] = astro_norm * 1.e-18 * df['I3MCWeightDict_OneWeight'] / (df['I3MCWeightDict_NEvents'] * nfiles) * np.power(df['I3MCWeightDict_PrimaryNeutrinoEnergy']/1.e5, astro_index)* ltime
    
    df['conv_weight'] = df['I3MCWeightDict_OneWeight'] * 2. / (df['I3MCWeightDict_NEvents'] * nfiles) * (df[f"cscdSBU_{cosmicray}_CORSIKA_SouthPole_June_{hadronicinteraction}_conv"]+df[f"cscdSBU_{cosmicray}_CORSIKA_SouthPole_December_{hadronicinteraction}_conv"])/2  * df[f'cscdSBU_HillasGaisser2012_H4a_CORSIKA_SouthPole_SIBYLL2.3c_conv_{sv}_passing_rate'] * ltime

    
    df['prompt_weight'] = df['I3MCWeightDict_OneWeight'] * 2. / (df['I3MCWeightDict_NEvents'] * nfiles) * (df[f"cscdSBU_{cosmicray}_CORSIKA_SouthPole_June_SIBYLL2.3c_pr"]+df[f"cscdSBU_{cosmicray}_CORSIKA_SouthPole_December_SIBYLL2.3c_pr"])/2 * df[f'cscdSBU_HillasGaisser2012_H4a_CORSIKA_SouthPole_SIBYLL2.3c_pr_{sv}_passing_rate'] * ltime

    df['weight'] = df['astro_weight']+df['conv_weight']+ df['prompt_weight']



def add_pass1_nugen_weights(df, ltime,indir):    
    astro_norm = 1.6
    astro_index = -2.5

    print(indir)
    nfiles = len(glob.glob(indir+f"/00*/final_cascade/*.i3.*"))
    if nfiles == 0:
        nfiles = len(glob.glob(indir+f"/p0=0.0_p1=0.0_domeff=1.00/00*/final_cascade/*.i3.*"))
    print('number of files ',nfiles)
    df['nfiles'] = np.ones(len(df.index))*nfiles

    
    df['astro_weight'] = astro_norm * 1.e-18 * df['I3MCWeightDict_OneWeight'] / (df['I3MCWeightDict_NEvents'] * nfiles) * np.power(df['I3MCWeightDict_PrimaryNeutrinoEnergy']/1.e5, astro_index) * ltime
    df['conv_weight'] = df['I3MCWeightDict_OneWeight'] * 2. / (df['I3MCWeightDict_NEvents'] * nfiles) * df['cscdSBU_AtmWeight_Conv_value'] * df['cscdSBU_AtmWeight_Conv_PassRate_value'] * ltime
    df['prompt_weight'] = df['I3MCWeightDict_OneWeight'] * 2. / (df['I3MCWeightDict_NEvents'] * nfiles) * df['cscdSBU_AtmWeight_Prompt_value'] * df['cscdSBU_AtmWeight_Prompt_PassRate_value'] * ltime / 2.

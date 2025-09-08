import numpy as np
import pandas as pd

def annotate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with
      - df['I3MCWeightDict_InIceNeutrinoType']  in {12,14,16}  (ν_e, ν_μ, ν_τ)
      - df['I3MCWeightDict_InteractionType']   in {1,2}       (1=CC, 2=NC) for ν_μ only
    this adds:
      - df['flavor']      in {'nue','numu','nutau'}
      - df['is_nue'], df['is_numu'], df['is_nutau']  (bool)
      - df['is_numuCC'], df['is_numuNC']             (bool)
      - df['sig_bdt1'], df['bg_bdt1']  (bool)  
           for BDT1:   signal = ν_τ;  background = ν_e ∪ ν_μ(NC)
      - df['sig_bdt2'], df['bg_bdt2']  (bool)  
           for BDT2:   signal = ν_τ;  background = ν_μ(CC)
    """
    # map integer ptype → string
    pmap = {12: "nue", 14: "numu", 16: "nutau"}
    df = df.copy()
    df["flavor"]  = abs(df["I3MCWeightDict_InIceNeutrinoType"]).map(pmap)
    
    # basic flavor flags
    df["is_nue"]    = df["flavor"] == "nue"
    df["is_numu"]   = df["flavor"] == "numu"
    df["is_nutau"]  = df["flavor"] == "nutau"
    
    # separate CC vs NC for numu
    df["is_numuCC"] = df["is_numu"] & (df["I3MCWeightDict_InteractionType"] == 1)
    df["is_numuNC"] = df["is_numu"] & (df["I3MCWeightDict_InteractionType"] == 2)
    
    # BDT1: ντ vs {νe + νμ(NC)}
    df["sig_bdt1"] = df["is_nutau"]
    df["bg_bdt1"]  = df["is_nue"] | df["is_numuNC"]
    
    # BDT2: ντ vs {νμ(CC)}
    df["sig_bdt2"] = df["is_nutau"]
    df["bg_bdt2"]  = df["is_numuCC"]
    
    return df


def create_learning_input(df, features):
    df_learn = pd.DataFrame()
    for key in features:
        df_learn[key] = df[key]
    df_learn.index = df.index
    return df_learn


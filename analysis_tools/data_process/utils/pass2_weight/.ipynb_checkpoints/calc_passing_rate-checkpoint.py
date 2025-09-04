#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.0.1/icetray-start
#METAPROJECT: /cvmfs/icecube.opensciencegrid.org/users/zzhang1/combo_stable_addatmo/

from icecube import MuonGun
from icecube import photospline
from icecube import icetray,dataio,dataclasses,hdfwriter
from I3Tray import *

import numpy as np
import pandas as pd

from argparse import ArgumentParser
import os

ptypes = {
     12:"nu_e",
     -12:"nu_e",
     14:"nu_mu",
     -14:"nu_mu",
     16:"nu_tau",
     -16:"nu_tau",
     }

def get_pr(df,pmodel,density,hadr,decay,ptype,outkey,plight):
    mcprimary = pd.DataFrame()
    mcprimary["energy"] = df["cscdSBU_MCPrimary_energy"]
    mcprimary["zenith"] = df["cscdSBU_MCPrimary_zenith"]
    mcprimary["azimuth"] = df["cscdSBU_MCPrimary_azimuth"]
    mcprimary["x"] = df["cscdSBU_MCPrimary_x"]
    mcprimary["y"] = df["cscdSBU_MCPrimary_y"]
    mcprimary["z"] = df["cscdSBU_MCPrimary_z"]
    mcprimary["pdg_encoding"] = df["cscdSBU_MCPrimary_type"]
    log_enu = np.log10(mcprimary.energy)
    ct = np.cos(mcprimary.zenith)

    gcd="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz"
    surface = MuonGun.ExtrudedPolygon.from_file(gcd)
    mcprimary["I3Position"] = [dataclasses.I3Position(x,y,z) for x,y,z in zip(mcprimary.x,mcprimary.y,mcprimary.z)]
    mcprimary["I3Direction"] = [dataclasses.I3Direction(x,y) for x,y in zip(mcprimary.zenith,mcprimary.azimuth)]
    d = [surface.intersection(pos,direction) for pos,direction in zip(mcprimary.I3Position,mcprimary.I3Direction)]
    getDepth = [pos+d.first*direction for pos,d,direction in zip(mcprimary.I3Position,d,mcprimary.I3Direction)]
    impactDepth = [MuonGun.depth(getdepth.z)*1.e3 for getdepth in getDepth]
    pdg_encoding = mcprimary.pdg_encoding.astype(int)
    ptype = ptypes[list(pdg_encoding)[0]]

    spline_name = f'/data/user/zzhang1/generate_spline/final_fits_{plight}/pr_{pmodel}_{density}_{hadr}_{decay}_{ptype}.fits'
    if not os.path.exists(spline_name):
        print("spline file does not exist:",spline_name)
        return False
    spline = photospline.I3SplineTable(spline_name)
    pr_p = [float(np.where(pdg_encoding>0,np.where(ct>0.05,spline.eval([log_enu,ct,impactDepth]),1),1)) for log_enu,ct,impactDepth,pdg_encoding in zip(log_enu,ct,impactDepth,pdg_encoding)]
    pr_p = [pr if pr>=0 else 0 for pr in pr_p]
    pr_p = [pr if pr<=1 else 1 for pr in pr_p]

    spline_name_antip = f'/data/user/zzhang1/generate_spline/final_fits_{plight}/pr_{pmodel}_{density}_{hadr}_{decay}_{ptype}bar.fits'
    spline_antip = photospline.I3SplineTable(spline_name_antip)
    pr_antip = [float(np.where(pdg_encoding<0,np.where(ct>0.05,spline_antip.eval([log_enu,ct,impactDepth]),1),1)) for log_enu,ct,impactDepth,pdg_encoding in zip(log_enu,ct,impactDepth,pdg_encoding)]
    pr_antip = [pr if pr>=0 else 0 for pr in pr_antip]
    pr_antip = [pr if pr<=1 else 1 for pr in pr_antip]
    pr = [pr_p*pr_antip for pr_p,pr_antip in zip(pr_p,pr_antip)]
    df[outkey] = pr
    return df



def calc_passing_rate(df):

    pdg_encoding = df["cscdSBU_MCPrimary_type"]
    ptype = ptypes[int(list(pdg_encoding)[0])]
    #plights = ["step5","step100","step250","step340","step500","step750","step1000","step1500","step2000","step2500","step3000"]
    plights = ["step340"]
    hadrs = ["SIBYLL2.3c"]
    pmodels = ["HillasGaisser2012_H4a"]
    densities = ["CORSIKA_SouthPole_December","CORSIKA_SouthPole_June"]
    for pmodel in pmodels:
        for hadr in hadrs:
            for plight in plights:
                for density in densities:
                    if ptype == "nu_tau":
                        decay = "conv"
                        outkey = f"cscdSBU_{pmodel}_{density}_{hadr}_{decay}_{plight}_passing_rate"
                        df[outkey] = 1
                        decay = "pr"
                        outkey = f"cscdSBU_{pmodel}_{density}_{hadr}_{decay}_{plight}_passing_rate"
                        df[outkey] = 1
                        continue

                    if hadr in ['SIBYLL2.3c', 'SIBYLL2.3','QGSJET-II-04', 'EPOS-LHC']:
                        decay = "conv"
                        outkey = f"cscdSBU_{pmodel}_{density}_{hadr}_{decay}_{plight}_passing_rate"
                        df = get_pr(df,pmodel,density,hadr,decay,ptype,outkey,plight)
                    
                    if hadr in ['SIBYLL2.3c', 'SIBYLL2.3', 'DPMJET-III-3.0.6']:
                        decay = "pr"
                        outkey = f"cscdSBU_{pmodel}_{density}_{hadr}_{decay}_{plight}_passing_rate"
                        df = get_pr(df,pmodel,density,hadr,decay,ptype,outkey,plight)
                for decay in ["conv","pr"]:
                    outkey = f"cscdSBU_{pmodel}_CORSIKA_SouthPole_{hadr}_{decay}_{plight}_passing_rate"
                    outkey_jun = f"cscdSBU_{pmodel}_CORSIKA_SouthPole_June_{hadr}_{decay}_{plight}_passing_rate"
                    outkey_dec = f"cscdSBU_{pmodel}_CORSIKA_SouthPole_December_{hadr}_{decay}_{plight}_passing_rate"
                    df[outkey] = (df[outkey_jun]+df[outkey_dec])/2
    return df

def main():
    parser = ArgumentParser()
    parser.add_argument("-o", "--out", type=str,default="", dest="OUT",help="out")
    parser.add_argument("-i", "--infile", dest="INFILE", default="", type=str, help="infile")
    args = parser.parse_args()
    outfile = args.OUT
    infile = args.INFILE
    df = pd.read_hdf(infile)
    calc_passing_rate(df)
    df.to_hdf(outfile,"hdf",mode="w")
    
if __name__ == "__main__":
    main()

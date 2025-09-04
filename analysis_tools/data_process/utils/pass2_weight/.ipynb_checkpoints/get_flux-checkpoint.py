#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy import interpolate
from functools import lru_cache

@lru_cache(2**12)
def create_spline(pmodel="HillasGaisser2012_H3a", density="CORSIKA_SouthPole_June", hadr="SIBYLL2.3c", kind="conv_nue"):
    """
    Parameterize the conventional and prompt neutrino flux as a function of
    neutrino energy and zenith angle.
    """
    infilename=f"/data/user/zzhang1/pass2_GlobalFit/mceq/generate_spline/h5/{pmodel}_{density}_{hadr}.h5"
    tempdf = pd.read_hdf(infilename,f"{kind}")
    pivoted = tempdf.pivot(index="energy",columns="theta_deg")
    enu = list(pivoted.index)
    theta_deg = list(pivoted.columns.get_level_values(1))
    enu_g, theta_deg_g = list(map(np.transpose, np.meshgrid(enu, theta_deg)))

    flux = pivoted.flux.values
    f = interpolate.interp2d(theta_deg, enu, flux, kind='cubic')
    return f

pmodels = ["HillasGaisser2012_H4a","GaisserStanevTilav_4-gen","HillasGaisser2012_H3a"]
#pmodels = ["GaisserStanevTilav_4-gen"]
#pmodels = ["HillasGaisser2012_H4a"]
densities = ["CORSIKA_SouthPole_June","CORSIKA_SouthPole_December"]
hadrs = ['SIBYLL2.3c','DPMJET-III-19.1','EPOS-LHC','QGSJET-II-04']
#hadrs = ['DPMJET-III-19.1']
kinds = {
        12:"nue",
        -12:"antinue",
        14:"numu",
        -14:"antinumu",
        16:"nutau",
        -16:"antinutau"
        }

def get_flux(df):
    #df = pd.read_hdf(infile)
    pdg = df.cscdSBU_MCPrimary_type
    theta_deg = df.cscdSBU_MCPrimary_zenith*180/3.1416
    theta_deg = 90-np.abs(theta_deg-90)
    enu = df.cscdSBU_MCPrimary_energy
    def get_flux_local(df,pmodel,density,hadr):
        p = np.abs(pdg.values[0])
        kind = f"conv_{kinds[p]}"
        f_p = create_spline(pmodel, density, hadr, kind)
        kind = f"conv_{kinds[-p]}"
        f_pbar = create_spline(pmodel, density, hadr, kind)
        outkey = f"cscdSBU_{pmodel}_{density}_{hadr}_conv"
        flux_p = pd.DataFrame([f_p(theta_deg, enu) for theta_deg,enu in zip(theta_deg,enu)], columns=["value"])
        flux_p[pdg<0] = 0 
        flux_pbar = pd.DataFrame([f_pbar(theta_deg, enu) for theta_deg,enu in zip(theta_deg,enu)], columns=["value"])
        flux_pbar[pdg>0] = 0 
        flux = flux_p + flux_pbar
        df[outkey] = flux

        p = np.abs(pdg.values[0])
        kind = f"pr_{kinds[p]}"
        f_p = create_spline(pmodel, density, hadr, kind)
        kind = f"pr_{kinds[-p]}"
        f_pbar = create_spline(pmodel, density, hadr, kind)
        outkey = f"cscdSBU_{pmodel}_{density}_{hadr}_pr"
        flux_p = pd.DataFrame([f_p(theta_deg, enu) for theta_deg,enu in zip(theta_deg,enu)], columns=["value"])
        flux_p[pdg<0] = 0 
        flux_pbar = pd.DataFrame([f_pbar(theta_deg, enu) for theta_deg,enu in zip(theta_deg,enu)], columns=["value"])
        flux_pbar[pdg>0] = 0 
        flux = flux_p + flux_pbar
        df[outkey] = flux
        return df

    for density in densities:
        for pmodel in pmodels:
            hadr = 'SIBYLL2.3c'
            df = get_flux_local(df,pmodel=pmodel,density=density,hadr=hadr)
        for hadr in hadrs:
            pmodel = 'HillasGaisser2012_H4a'
            df = get_flux_local(df,pmodel=pmodel,density=density,hadr=hadr)

    #df.to_hdf(outfile,"hdf",mode="w")
    return df    


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-o", "--out", type=str,default="", dest="OUT",help="out")
    parser.add_argument("-i", "--infile", dest="INFILE", default="", type=str, help="infile")
    args = parser.parse_args()
    outfile = args.OUT
    infile = args.INFILE
    df = pd.read_hdf(infile)
    get_flux(df)
    df.to_hdf(outfile,"hdf",mode="w")

if __name__ == "__main__":
    main()

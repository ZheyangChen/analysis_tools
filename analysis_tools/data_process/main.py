import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from utils.pass2_weight.get_flux import get_flux
from utils.pass2_weight.calc_passing_rate import calc_passing_rate
from utils.pass2_weight.calculate_weight import add_nugen_weights,add_pass1_nugen_weights
from precuts.precuts import precuts,singlekey_cut
from utils.add_variables.add_variables import add_variables


def main():
    parser = argparse.ArgumentParser(description="Data Processing Toolkit")
    #parser.add_argument('-c', '--config', help="Config file path")
    parser.add_argument('-i', '--input', help="Override input path")
    parser.add_argument('-o', '--output', help="Override output path")
    parser.add_argument('-flux', '--pass2_flux', action='store_true', default=False, help='calculate pass2 flux')
    parser.add_argument('-add_weight', '--add_weight', help="add mc weights, type indir of mc datasets to calculate nfiles for weight")
    parser.add_argument('-add_pass1_weight', '--add_pass1_weight', help="add mc weights, type indir of mc datasets to calculate nfiles for weight")

    parser.add_argument('-ecut', '--energycut', help='Select events with energy larger than a certain value')
    parser.add_argument('-precut', '--precut', action='store_true', default=False, help='apply precuts to dataframe')
    parser.add_argument('-add_var','--add_variable', action='store_true', default=False, help='calculate some BDT variables for some old files')

    args = parser.parse_args()

    if args.input:
        infile = args.input
        df = pd.read_hdf(infile)
        print('Infile: ',infile)
    if args.output:
        outfile = args.output
        print('Outfile: ',outfile)

    if args.energycut:
        print('applying energy cut')
        cut_column = 'cscdSBU_MonopodFit4_noDC_energy'
        cut_value = 10**(float(args.energycut))
        df = singlekey_cut(df , cut_column , [(">", cut_value)])

    if args.precut:
        print('applying precuts')
        df = precuts(df)

    if df.empty:
        print('Dataframe is empty')
    else:
        print('Dataframe not empty after cuts')
        if args.add_variable:
            print('Calculating some BDT variables')
            add_variables(df)
    
        if args.pass2_flux:
            print('calculating pass2 flux')
            get_flux(df)
            calc_passing_rate(df)
    
        if args.add_weight:
            ltime = 3600*24*365.
            add_nugen_weights(df, ltime,args.add_weight)
        elif args.add_pass1_weight:
            ltime = 3600*24*365.
            add_pass1_nugen_weights(df, ltime,args.add_pass1_weight)
            
    
        
    df.to_hdf(outfile,key = 'hdf',mode = 'w')



if __name__ == "__main__":
    main()

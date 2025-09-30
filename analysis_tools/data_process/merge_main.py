import argparse
import numpy as np
import pandas as pd
import glob
import os
import h5py
import re
from pathlib import Path
from merge.singlekey_merge import merge,split_hdf_key
from converti3toh5.converti3toh5 import converti3toh5


def main():
    parser = argparse.ArgumentParser(description="Data Processing Toolkit")
    #parser.add_argument('-c', '--config', help="Config file path")
    parser.add_argument('-i', '--input', help="Override input path")
    parser.add_argument('-o', '--output', help="Override output path")
    parser.add_argument('-id', '--simprodid', help="simprod id")
    parser.add_argument('-convert', '--i3convert', action='store_true', default=False,help = 'whether to convert i3 to h5')
    #parser.add_argument('-split_key', '--split_key', action='store_true', default=False,help = 'used to split taupede particle keys')
    parser.add_argument('-split_key', '--split_key',help = 'type name of the key to split, used to split taupede particle keys')
    parser.add_argument('-m', '--merge', action='store_true', default=False,  help="runing single key merging code")
    parser.add_argument('-isdata', action='store_true', default=False,  help="If this is running on data")
    parser.add_argument('-ismgun', action='store_true', default=False,  help="If this is running for muon gun")


    args = parser.parse_args()
    
    if args.input:
        indir_master = args.input
        print('indir: ',indir_master)

    
    if args.output:
        outdir = args.output
        print('outdir: ',outdir)
        try:
            os.stat(outdir)
        except:
            print(f"making dir {outdir}")
            os.makedirs(outdir)
  

    if args.simprodid:
        simprodid = args.simprodid
        print('simprodid is: ',simprodid)

    if args.ismgun:
        print('Running for muongun files')
    if args.isdata:
        print('Running for real data')

    if args.i3convert:  
        print('convert i3 files to hdf files')
        if not args.simprodid:
            print('processing single directory')
            indir = indir_master
            filelist = sorted(glob.glob(indir+"/*evel*.i3.*"))
            print('filelist: ',filelist)
            for infile in filelist:
                try:
                    filename = infile.split('/')[-1]
                    print('filename: ',filename)
                    outdfinal = indir + '/h5file/'
                    try:
                        os.stat(outdfinal)
                    except:
                        print(f"making dir {outdfinal}")
                        os.makedirs(outdfinal)
                        
                    outfilefinal = outdfinal+infile.split('/')[-1][:-7]+'.h5'
                    converti3toh5(infile,outfilefinal,args.ismgun,args.isdata)
                    #print('processing ',filename)
                except:
                    print('error runing hdf writer')
                if args.split_key:
                    print('spliting keys')
                    try:
                        #split_hdf_key(hdf_path=outfilefinal,source_key='Taupede_spice3Particles',replace_name = 'Taupede',new_key_suffix=('Taupede1', 'Taupede2'))
                        key_name = args.split_key
                        split_hdf_key(hdf_path=outfilefinal,source_key=f'{key_name}Particles',replace_name = f'{key_name}',new_key_suffix=(f'{key_name}_1', f'{key_name}_2'))
                    except:
                        pass
        else:
            print(f'processing {simprodid}')
            indir = sorted(glob.glob(indir_master+f"/{simprodid}/00*/final_cascade/"))
            #indir = sorted(glob.glob(indir_master+f"/{simprodid}/p0=0.0_p1=0.0_domeff=1.00/00*/final_cascade/"))
            outdir_final = [(i+'h5file/') for i in indir]
            
            for ind,outdfinal in zip(indir,outdir_final):
                filelist = sorted(glob.glob(ind+"/*evel*.0*.i3.*"))
                try:
                    os.stat(outdfinal)
                except:
                    print(f"making dir {outdfinal}")
                    os.makedirs(outdfinal)
                for infile in filelist:
                    outfilefinal = outdfinal+infile.split('/')[-1][:-7]+'.h5'
                    converti3toh5(infile,outfilefinal,args.ismgun,args.isdata)
                    if args.split_key:
                        print('spliting keys')
                        try:
                            #split_hdf_key(hdf_path=outfilefinal,source_key='Taupede_spice3Particles',replace_name = 'Taupede',new_key_suffix=('Taupede1', 'Taupede2'))
                            key_name = args.split_key
                            split_hdf_key(hdf_path=outfilefinal,source_key=f'{key_name}Particles',replace_name = f'{key_name}',new_key_suffix=(f'{key_name}_1', f'{key_name}_2'))
                        except:
                            pass
                        

        
        
    
    if args.merge:
        count = 0
        out_set = pd.DataFrame()
        print('start merging files')
        if args.simprodid:
            #indir = sorted(glob.glob(indir_master+f"/{simprodid}/00*/final_cascade/h5file/"))
            indir = sorted(glob.glob(indir_master+f"/{simprodid}/*h5file/"))
            #indir = sorted(glob.glob(indir_master+f"/{simprodid}/p0=0.0_p1=0.0_domeff=1.00/00*/final_cascade/h5file/"))
            #indir = sorted(glob.glob(indir_master+f"/{simprodid}/0043000-0043999/final_cascade/h5file/"))
            for ind in indir:
                #print('current indir is: ',ind)
                filelist = sorted(glob.glob(ind+"*evel*.0*.h5"))
                print('Merging {len(filelist)} files')
                for infile in filelist:
                    try:
                        f = h5py.File(infile)
                        #print('current infile is:',infile)
                        df_add = merge(infile,list(f.keys()))
                        print(infile)
                        out_set = pd.concat([out_set,df_add],ignore_index = True)
                        count+=1
                        print(infile,' Done')
    
                    except:
                        pass
            out_set.to_hdf(outdir +'/'+ 'Finallevel_nugen_'+str(simprodid)+'.h5',key='h',mode='w')
        else:
            count = 0
            out_set = pd.DataFrame()
            print('start merging files')
            if args.i3convert:
                indir = outdfinal
            else:
                indir = indir_master
            filelist = sorted(glob.glob(indir+"/"+"*.h5"))
            for infile in filelist:
                try:
                    f = h5py.File(infile)
                    #print('current infile is:',infile)
                    df_add = merge(infile,list(f.keys()))
                    print(infile)
                    out_set = pd.concat([out_set,df_add],ignore_index = True)
                    count+=1
                    print(infile,' Done')

                except:
                    pass
            if args.isdata:
                pattern_year = r'.*/(IC86_[^/]+)/.*'
                match = re.search(pattern_year, indir)
                if match:
                    ICyear = match.group(1)
                else:
                    ICyear = 'IC86'
                    print("Don't know IC year.")
                    
                pattern_burnorblind = r".*/(burn|blind)/.*"
                match = re.search(pattern_burnorblind, indir)
                
                if match:
                    burnorblind = match.group(1)  # This will be either "nameA" or "nameB"
                    print("burn or blind data:", burnorblind)
                else:
                    # look for any path component beginning with "Run"
                    parts    = indir.split(os.path.sep)
                    run_comp = next((p for p in parts if p.startswith("Run")), None)
                    if run_comp:
                        burnorblind = run_comp
                        print("Found run component:", burnorblind)
                    else:
                        burnorblind = 'data'
                        print("Don't know burn/blind or Run, defaulting to:", burnorblind)
                out_set['ICyear'] = ICyear[-4:]
                out_set.to_hdf(outdir +'/'+ f'Finallevel_{ICyear}_{burnorblind}.h5',key='h',mode='w')
            else:
                out_set.to_hdf(outdir +'/'+ 'Finallevel_nugen_'+indir_master.split('/')[-1]+'.h5',key='h',mode='w')


    


    









main()

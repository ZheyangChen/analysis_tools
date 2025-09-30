import subprocess
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser(
        description="Generate a list of values (int) and then convert to strings."
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
        "-r", "--range",
        nargs=2, type=int, metavar=("START", "END"),
        help="Start and end (inclusive) for a continuous range"
)
group.add_argument(
        "-v", "--values",
        nargs="+", type=int, metavar="N",
        help="One or more discrete integer values"
)
parser.add_argument(
        "-i",
        "--indir",
        type=Path,
        default=Path("/data/ana/analyses/diffuse/cascades-nutau/sim/nugen/taupede/snowstorm/"),
        help="directory to read HDF files from (default: %(default)s)"
    )

parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("/data/ana/analyses/diffuse/cascades-nutau/sim/nugen/taupede/snowstorm/combined_datasets/single_key_files/"),
        help="path to output directory (default: %(default)s)"
    )

parser.add_argument(
        "-m",
        "--ismgun",
        action='store_true', default=False,  help="If this is running for muon gun")

parser.add_argument(
        "-d",
        "--isdata",
        action='store_true', default=False,  help="If this is running for data")

args = parser.parse_args()

#indir_master = '/data/user/zchen/pass2_cascade_nutau/taupede/file/data/'
indir = args.indir
print(f"indir is {indir}")
outdir = args.outdir
print(f"outdir is {outdir}")

if args.range:
    start, end = args.range
    id_list = [str(prodid) for prodid in range(start, end + 1)]
elif args.values:
    values = args.values
    id_list = [str(v) for v in values]
else:
    print("no value provided")
    

print(id_list)
#Taupede_key = 'Taupede_spice3'

for id in id_list:
    print('processing merging ',str(id))
    if args.ismgun:
        subprocess.run(['python','merge_main.py','-i',indir,'-o',outdir,'-convert','-id',id,'-m','-ismgun'])
    elif args.isdata:
        indir_year = f'{indir}/IC86_{id}/burn/final_cascade/'
        print(f'Running code for {indir_year}')
        subprocess.run(['python','merge_main.py','-i',indir_year,'-o',outdir,'-convert','-m','-isdata'])
    else:
        #subprocess.run(['python','merge_main.py','-i',indir,'-o',outdir,'-convert','-id',id,'-m'])
        subprocess.run(['python','merge_main.py','-i',indir,'-o',outdir,'-id',id,'-m'])
    #subprocess.run(['python','merge_main.py','-i',indir,'-o',outdir,'-id',str(id),'-convert','-split_key','-m'])

# batch_run.py
import subprocess
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import re
import time



def batch_process():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-o", "--output-dir", default="batch_output")
    parser.add_argument("-j", "--workers", type=int, default=4)
    
    # Add range/value selection like in merge_grouprun.py
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-r", "--range",
        nargs=2, type=int, metavar=("START", "END"),
        help="Start and end (inclusive) for a continuous range of simprod IDs"
    )
    group.add_argument(
        "-v", "--values",
        nargs="+", type=int, metavar="N",
        help="One or more discrete simprod ID values"
    )
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    outdir = str(output_dir)

    all_files = [
        f for f in input_dir.glob("*")
        if f.is_file() and f.suffix.lower() in {'.h5', '.hdf'}
    ]

    # Filter files based on range/value arguments
    if args.range or args.values:
        if args.range:
            start, end = args.range
            target_ids = set(str(prodid) for prodid in range(start, end + 1))
        else:  # args.values
            target_ids = set(str(v) for v in args.values)
        
        print(f"Filtering files for simprod IDs: {sorted(target_ids)}")
        
        # Filter files that match the target simprod IDs
        filtered_files = []
        for file in all_files:
            file_name = file.name
            pattern = r"(\d{5})(?=\.[^.]+$)"
            match = re.search(pattern, file_name)
            if match:
                simid = match.group(1)
                if simid in target_ids:
                    filtered_files.append(file)
        
        all_files = filtered_files
        print(f"Found {len(all_files)} files matching the criteria")

    Failed_list = []
    for file in all_files:
        infile = str(file)
        file_name = infile.split('/')[-1]
        pattern = r"(\d{5})(?=\.[^.]+$)"
        simid = (re.search(pattern, file_name)).group(1)
        #simid = infile[-8:-3]
        spice3_nonsnowtorm_path = '/data/ana/analyses/diffuse/cascades-nutau/sim/nugen/taupede/' + simid
        spice3_path = '/data/ana/analyses/diffuse/cascades-nutau/sim/nugen/taupede/snowstorm/' + simid
        ftp_path = '/data/ana/analyses/diffuse/cascades-nutau/sim/nugen/taupede/snowstorm/ftp_reco/' + simid
        i3_path = (
            spice3_nonsnowtorm_path if Path(spice3_nonsnowtorm_path).exists()
            else spice3_path if Path(spice3_path).exists()
            else ftp_path if Path(ftp_path).exists()
            else None
        )
        if i3_path is None:
           raise FileNotFoundError("None of the candidate paths exist")
        print('outfile is ',file_name)
        print(i3_path)
        start_time = time.time()
        try:
            #subprocess.run(['python','main.py','-i',infile,'-o',outdir+'/'+file_name,'-add_var','-flux','-add_weight',i3_path])
            subprocess.run(['python','main.py','-i',infile,'-o',outdir+'/'+file_name,'-add_var','-flux','-add_weight',i3_path],check=True)
            #subprocess.run(['python','main.py','-i',infile,'-o',outdir+'/'+file_name,'-add_var','-flux','-add_weight',i3_path,'-ecut','4.5'],check=True)
            #subprocess.run(['python','main.py','-i',infile,'-o',outdir+'/'+file_name,'-add_var','-flux','-add_weight',i3_path,'-precut'])
            #subprocess.run(['python','main.py','-i',infile,'-o',outdir+'/'+file_name,'-add_var','-add_pass1_weight',i3_path,'-precut'])
            elapsed = time.time() - start_time
            print(f"  ✅ Attempt {file_name}: Success ({elapsed:.1f}s)")
        except:
            Failed_list.append(file_name)
            pass
        '''
        except subprocess.CalledProcessError as e:
            Failed_list.append(file_name)
            elapsed = time.time() - start_time
            error_info = {
                'returncode': e.returncode,
                'output': e.stdout,
                'error': e.stderr,
                'time': elapsed
            }
            
            print(f"  ⚠️ Attempt {file_name}: Failed (code {e.returncode}, {elapsed:.1f}s)")
            print(f"     Error: {e.stderr[:100]}...") 

        except subprocess.TimeoutExpired:
                Failed_list.append(file_name)
                error_info = {
                    'error': f"Timeout after {timeout}s",
                    'time': timeout
                }
                print(f"  ⏰ Attempt {file_name}: Timeout")
    '''

    print('Failed jobs: ', Failed_list)
            
'''
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for file_path in all_files:
            future = executor.submit(process_file, file_path, output_dir)
            futures.append(future)
        
        for future in futures:
            try:
                future.result()  
            except Exception as e:
                print(f"process failed: {str(e)}")
'''



if __name__ == "__main__":
    batch_process()


import os
import glob
import re
import pandas as pd
from typing import List, Dict, Union, Optional
from typing_extensions import Literal  # for older Python versions
from pathlib import Path


def find_data_files(
    df: pd.DataFrame,
    master_dir: str,
    run_col: str = "I3EventHeader_Run",
    file_extensions: Union[str, List[str]] = [".i3.zst", ".i3.bz2", ".i3"],
    return_as: Literal["dataframe", "dict"] = "dataframe",
    select: Literal["first", "last", "mtime_newest", "mtime_oldest"] = "first",
    recursive: bool = True
) -> Union[pd.DataFrame, Dict[int, List[str]]]:
    """
    Locate IceCube data files by matching 6-digit run numbers in file names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a column with run numbers.
    master_dir : str
        Root directory to search for files.
    run_col : str
        Column name containing 6-digit run numbers.
    file_extensions : str or list
        Allowed file extensions to match (e.g., .i3.zst).
    return_as : 'dataframe' or 'dict'
        Whether to return a DataFrame or dict of matches.
    select : 'first' | 'last' | 'mtime_newest' | 'mtime_oldest'
        How to choose one file from multiple matches.
    recursive : bool
        If True, will search recursively under master_dir.

    Returns
    -------
    pd.DataFrame or dict
        DataFrame with match info, or dict of {run_id: [matching files]}.
    """
    if isinstance(file_extensions, str):
        file_extensions = [file_extensions]

    if run_col not in df.columns:
        raise ValueError(f"DataFrame must have a '{run_col}' column")

    out_rows = []
    out_dict = {}

    def _choose(paths: List[str]) -> Optional[str]:
        if not paths:
            return None
        if select == "first":
            return paths[0]
        if select == "last":
            return paths[-1]
        if select == "mtime_newest":
            return max(paths, key=os.path.getmtime)
        if select == "mtime_oldest":
            return min(paths, key=os.path.getmtime)
        return paths[0]

    for _, row in df.iterrows():
        run = row[run_col]
        try:
            run_id = int(run)
            if not (100000 <= run_id <= 999999):
                raise ValueError
        except:
            run_id = None

        if run_id is None:
            matches = []
        else:
            pattern = f"**/*{run_id:06d}*"
            all_matches = glob.glob(
                os.path.join(master_dir, pattern),
                recursive=recursive
            )
            matches = sorted([
                path for path in all_matches
                if any(path.endswith(ext) for ext in file_extensions)
            ])

        if return_as == "dict":
            out_dict[run_id] = matches
        else:
            out_rows.append({
                **row.to_dict(),
                "run_id": run_id,
                "matches": matches,
                "i3_path": _choose(matches)
            })

    return out_dict if return_as == "dict" else pd.DataFrame(out_rows)

def find_mc_files(
    df: pd.DataFrame,
    master_dir: str,
    run_col: str = "I3EventHeader_Run",
    file_extensions: Union[str, List[str]] = [".i3.zst", ".i3.bz2", ".i3"],
    return_as: Literal["dataframe", "dict"] = "dataframe",
    select: Literal["first", "last", "mtime_newest", "mtime_oldest"] = "first",
    recursive: bool = True,
) -> Union[pd.DataFrame, Dict[int, List[str]]]:
    """
    Locate IceCube MC files using 10-digit run numbers.

    Directory structure: {master_dir}/{simid}/{subid}/{category}/files
    """

    if isinstance(file_extensions, str):
        file_extensions = [file_extensions]

    if run_col not in df.columns:
        raise ValueError(f"DataFrame must have a '{run_col}' column")

    out_rows = []
    out_dict = {}

    run_cache = {}

    def _choose(paths: List[str]) -> Optional[str]:
        if not paths:
            return None
        if select == "first":
            return paths[0]
        if select == "last":
            return paths[-1]
        if select == "mtime_newest":
            return max(paths, key=os.path.getmtime)
        if select == "mtime_oldest":
            return min(paths, key=os.path.getmtime)
        return paths[0]

    for _, row in df.iterrows():
        run = row[run_col]

        try:
            run_id = int(run)
            run_str = f"{run_id:010d}"
        except Exception:
            run_id = None
            run_str = None

        if run_id in run_cache:
            matches = run_cache[run_id]
            simid = run_str[:5] if run_str else None
        else:
            if run_str is None or len(run_str) != 10:
                matches = []
                simid = None
            else:
                simid = run_str[:5]
                subid = run_str[5:]

                base_dir = os.path.join(master_dir, simid)
                if not os.path.isdir(base_dir):
                    matches = []
                else:
                    pattern = f"**/*0{simid}.0{subid}*"
                    all_matches = glob.glob(
                        os.path.join(base_dir, pattern),
                        recursive=recursive
                    )
                    matches = sorted([
                        p for p in all_matches
                        if any(p.endswith(ext) for ext in file_extensions)
                    ])
            run_cache[run_id] = matches

        if return_as == "dict":
            out_dict[run_id] = matches
        else:
            out_rows.append({
                **row.to_dict(),
                "run_id": run_id,
                "simid": simid,
                "matches": matches,
                "i3_path": _choose(matches),
            })

    return out_dict if return_as == "dict" else pd.DataFrame(out_rows)

def find_i3_files(df, master_dir, mode="data", **kwargs):
    if mode == "data":
        return find_data_files(df, master_dir, **kwargs)
    elif mode == "mc":
        return find_mc_files(df, master_dir, **kwargs)
    else:
        raise ValueError("mode must be 'data' or 'mc'")



GOODRUN_DIR = "/data/user/zchen/unblinding/goodrunlist/full_grllist/"

def load_goodrun_table(ICyear: int) -> pd.DataFrame:
    """
    Load the GoodRunInfo table for a given IC year.

    Returns:
        DataFrame with columns: ['runid', 'livetime', 'year', 'date']
    """
    ic_config = 79 if ICyear == 2010 else 86
    filename = f"IC{ic_config}_{ICyear}_GoodRunInfo_l3_final.txt"
    filepath = os.path.join(GOODRUN_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GoodRunInfo file not found: {filepath}")

    rows = []
    with open(filepath, 'r') as f:
        lines = f.readlines()[2:]  # skip first two header lines

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue  # skip malformed lines

        runid = parts[0]
        livetime = float(parts[3])
        full_path = parts[7]

        try:
            year = full_path.split('/')[4]
            date = full_path.split('/')[7]
        except IndexError:
            continue  # skip malformed paths

        rows.append({
            'runid': runid,
            'livetime': livetime,
            'year': year,
            'date': date
        })

    df = pd.DataFrame(rows)
    df['runid'] = df['runid'].astype(str)
    return df.set_index('runid')


def find_gcd_for_i3(i3_path: str) -> str:
    """
    Given an I3 file path, find the corresponding GCD file.

    Returns:
        Path to the matched GCD file (str) or raises FileNotFoundError.
    """
    filename = os.path.basename(i3_path)

    # Extract IC year: first match of 20xx_ not preceded by digit
    year_match = re.search(r'(?<!\d)(20\d{2})_', filename)
    if not year_match:
        raise ValueError(f"Could not extract IC year from filename: {filename}")
    ICyear = int(year_match.group(1))

    # Extract run ID: look for "Run00xxxxxx"
    run_match = re.search(r'Run00(\d{6})', filename)
    if not run_match:
        raise ValueError(f"Could not extract run ID from filename: {filename}")
    runid = run_match.group(1)

    # Load goodrun info for this year
    goodrun_df = load_goodrun_table(ICyear)

    if runid not in goodrun_df.index:
        raise ValueError(f"Run {runid} not found in GoodRunInfo for IC{ICyear}")

    row = goodrun_df.loc[runid]
    year = row["year"]
    date = row["date"]
    ic_config = 79 if ICyear == 2010 else 86

    # Build GCD glob pattern
    if ICyear == 2010:
        pattern = f"/data/exp/IceCube/{year}/filtered/level2pass2/{date}/Run00{runid}/Level2pass2_IC{ic_config}.{ICyear}_data_Run00{runid}*GCD*"
    elif ICyear >= 2017:
        pattern = f"/data/exp/IceCube/{year}/filtered/level2/{date}/Run00{runid}/Level2_IC86.{ICyear}_data_Run00{runid}*GCD*"
    else:
        pattern = f"/data/exp/IceCube/{year}/filtered/level2pass2/{date}/Run00{runid}/Level2pass2_IC{ic_config}.{ICyear}_data_Run00{runid}*GCD*"

    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No GCD file found for pattern:\n{pattern}")
    return matches[0]


'''
def find_files_for_runs(
    df: pd.DataFrame,
    master_dir: str,
    run_col: str = "I3EventHeader_Run",   # column with the run number
    zero_pad: int = 10,                   # abcdeVWXYZ -> 10 digits
    return_as: Literal["dataframe","dict"] = "dataframe",
    select: Literal["first","last","mtime_newest","mtime_oldest"] = "first",
) -> Union[pd.DataFrame, Dict[str, List[str]]]:
    """
    Attach file matches (and a chosen 'i3_path') for run IDs shaped like abcdeVWXYZ.

    Pattern searched:
      master_dir/{dataset}/00VW000-00VW999/*/*.0abcde.0VWXYZ*

    Returns
    -------
    - If return_as='dataframe': a copy of df with new columns:
        ['dataset','file','subrange','glob_pattern','matches','i3_path']
    - If return_as='dict': {run_str: [matches]}
    """
    if run_col not in df.columns:
        raise ValueError(f"DataFrame must have a '{run_col}' column")

    def _choose_path(paths: List[str]) -> Optional[str]:
        if not paths:
            return None
        if select == "first":
            return paths[0]
        if select == "last":
            return paths[-1]
        if select == "mtime_newest":
            return max(paths, key=lambda p: os.path.getmtime(p))
        if select == "mtime_oldest":
            return min(paths, key=lambda p: os.path.getmtime(p))
        return paths[0]

    out_rows = []
    out_map: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        run_val = row[run_col]

        # Normalize to 10-digit string
        if pd.isna(run_val):
            run_str = None
        elif isinstance(run_val, (int, float)) and not isinstance(run_val, bool):
            run_str = str(int(run_val)).zfill(zero_pad)
        else:
            s = str(run_val).strip()
            digits = "".join(ch for ch in s if ch.isdigit())
            run_str = digits.zfill(zero_pad) if digits else None

        if not run_str or len(run_str) < 10:
            dataset = file_id = VW = subrange = None
            glob_pattern = None
            matches: List[str] = []
        else:
            dataset = run_str[:5]         # abcde
            file_id = run_str[5:]         # VWXYZ
            VW      = file_id[:2]
            subrange = f"00{VW}000-00{VW}999"
            glob_pattern = os.path.join(
                str(master_dir).rstrip("/"),
                dataset,
                subrange,
                "*",
                f"*.0{dataset}.0{file_id}*"
            )
            matches = sorted(glob.glob(glob_pattern))

        if return_as == "dict":
            out_map[run_str or str(run_val)] = matches
        else:
            out_rows.append({
                **row.to_dict(),
                "dataset": dataset,
                "file": file_id,
                "subrange": subrange,
                "glob_pattern": glob_pattern,
                "matches": matches,
                "i3_path": _choose_path(matches)
            })

    if return_as == "dict":
        return out_map
    else:
        return pd.DataFrame(out_rows)
'''
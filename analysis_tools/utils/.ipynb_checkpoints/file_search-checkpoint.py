import os, glob
from typing import List, Dict, Union, Optional, Literal
import pandas as pd

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
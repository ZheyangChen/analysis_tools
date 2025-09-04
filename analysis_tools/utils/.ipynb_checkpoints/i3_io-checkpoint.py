# analysis_tools/utils/i3_io.py
import os, glob
from typing import Iterator, Optional, Tuple, Dict
import pandas as pd
from icecube import dataio, dataclasses
from icecube.icetray import I3Frame


def find_data_gcd(i3_path, data_base="/data/exp/IceCube/",goodrun_base="/data/user/zzhang1/pass2_GlobalFit/code/submission/goodrunlist/finalpass2"):
    """
    Given the path to a data i3 file, locate the matching GCD file.
    Does not assume any particular prefix before the ICxx.<year>_data_Run... part.
    """
    fname = os.path.basename(i3_path)
    # match IC<cfg>.<year> or IC<cfg>_<year>, then Run<runid>
    m = re.search(r"IC(\d+)[._](\d{4}).*Run*(\d+)", fname)
    if not m:
        raise ValueError(f"Could not parse IC, year, and run from '{fname}'")
    ic_cfg   = int(m.group(1))
    ICyear = int(m.group(2))
    runid_str = m.group(3)          # no leading zeros here
    runid_int = int(runid_str)      # numeric runID
    #run_padded = f"{int(runid):08d}"

    # pick filtered subdir based on ICyear
    if ICyear <= 2010:
        filtered_sub = "filtered/level2pass2"
    elif ICyear < 2017:
        filtered_sub = "filtered/level2pass2"
    else:
        filtered_sub = "filtered/level2"

    # read the good-run file for this IC-config & IC-year
    goodrun_file = os.path.join(
        goodrun_base,
        f"IC{ic_cfg}_{ICyear}_GoodRunInfo_l3_final.txt"
    )
    if not os.path.exists(goodrun_file):
        raise FileNotFoundError(f"GoodRunInfo not found: {goodrun_file}")

    
    year_dir = date_dir = None
    with open(goodrun_file) as f:
        lines = [l.split() for l in f]
    # header is two lines; actual data starts at index 2
    for row in lines[2:]:
        if int(row[0]) == runid_int:
            # row[7] looks like "/data/2018/filtered/.../<date>/Run1234567/..."
            parts = row[7].split('/')
            year_dir = parts[4]  # e.g. "2018"
            date_dir = parts[7]  # e.g. "2018-06-23"
            break
    #print('parts',parts)
    #print('year_dir',year_dir)
    #print('date_dir',date_dir)

    
    if year_dir is None:
        raise RuntimeError(f"Run {runid_int} not found in {goodrun_file}")

    # Now glob for any Run<runid_int> (no zero padding) under that path:
    pattern = os.path.join(
        data_base, year_dir, filtered_sub, date_dir,
        f"Run{runid_str}",             # EXACT un-padded run folder
        f"*IC{ic_cfg}.{ICyear}_*GCD*"
    )
    
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No GCD found for run {runid_int} under:\n  {pattern}")

    return matches[0]


def ensure_i3_paths(
    df: pd.DataFrame,
    master_dir: str,
    run_col: str = "I3EventHeader_Run",
) -> pd.DataFrame:
    """
    Return a copy of df with an 'i3_path' column resolved by find_files_for_runs.
    """
    from analysis_tools.utils.file_search import find_files_for_runs
    dfp = find_files_for_runs(df, master_dir, run_col=run_col, return_as="dataframe")
    if "i3_path" not in dfp.columns:
        raise RuntimeError("find_files_for_runs must return an 'i3_path' column")
    # align to original rows on index (safe merge on run if you prefer)
    out = df.copy()
    out = out.join(dfp[["i3_path"]], how="left")
    return out


def load_calibration(
    i3_path: str,
    *,
    gcd_mode: str = "mc",                  # "mc" or "data"
    gcd_file: Optional[str] = None,
) -> dataclasses.I3Calibration:
    """
    Load and return the I3Calibration frame for the given event file.
    """
    if gcd_file is None:
        if gcd_mode.lower() == "mc":
            base = "/cvmfs/icecube.opensciencegrid.org/data/GCD/"
            patt = os.path.join(base, "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz")
            found = sorted(glob.glob(patt))
            if not found:
                raise FileNotFoundError("No MC GCD found.")
            gcd_file = found[-1]
        else:
            gcd_file = find_data_gcd(i3_path)

    gf = dataio.I3File(gcd_file)
    cali = None
    try:
        while gf.more():
            fr = gf.pop_frame()
            if fr.Stop == I3Frame.Calibration and "I3Calibration" in fr:
                cali = fr["I3Calibration"]
                break
    finally:
        gf.close()

    if cali is None:
        raise RuntimeError(f"No I3Calibration in GCD: {gcd_file}")
    return cali


def open_physics_frame(
    i3_path: str,
    run_id: int,
    event_id: int,
    cali: dataclasses.I3Calibration,
):
    """
    Open the event's Physics frame, inject calibration, and return the frame.
    """
    f = dataio.I3File(i3_path)
    phys = None
    try:
        while f.more():
            fr = f.pop_physics()
            hdr = fr["I3EventHeader"]
            if hdr.run_id == run_id and hdr.event_id == event_id:
                fr.Put("I3Calibration", cali)
                phys = fr
                break
    finally:
        f.close()
    if phys is None:
        raise RuntimeError(f"No matching Physics frame: run={run_id}, event={event_id}")
    return phys


def iter_event_frames(
    df: pd.DataFrame,
    master_dir: str,
    *,
    run_col: str = "I3EventHeader_Run",
    event_col: str = "I3EventHeader_Event",
    gcd_mode: str = "mc",
    gcd_file: Optional[str] = None,
) -> Iterator[Dict]:
    """
    Yield dicts for each row with {run_id, event_id, i3_path, cali, phys}.
    Skips rows without a resolvable i3_path.
    """
    dfp = ensure_i3_paths(df, master_dir, run_col=run_col)
    for _, row in dfp[[run_col, event_col, "i3_path"]].iterrows():
        i3_path = row["i3_path"]
        if not isinstance(i3_path, (str, os.PathLike)) or pd.isna(i3_path):
            continue
        if not os.path.exists(i3_path):
            continue
        run_id = int(row[run_col])
        event_id = int(row[event_col])

        cali = load_calibration(i3_path, gcd_mode=gcd_mode, gcd_file=gcd_file)
        phys = open_physics_frame(i3_path, run_id, event_id, cali)

        yield {
            "run_id": run_id,
            "event_id": event_id,
            "i3_path": i3_path,
            "cali": cali,
            "phys": phys,
        }
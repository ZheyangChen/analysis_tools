import os
import re
import glob
from typing import Optional, Union, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from icecube import dataio, dataclasses, icetray
from icecube.icetray import OMKey, I3Frame





def _load_calibration(i3_path: str, gcd_mode: str = "mc", gcd_file: str | None = None):
    """
    Returns (cali, gcd_file). Uses find_data_gcd(i3_path) automatically for DATA
    when gcd_file is not provided.
    """
    if gcd_file is None:
        if gcd_mode == "mc":
            base = "/cvmfs/icecube.opensciencegrid.org/data/GCD/"
            patt = os.path.join(base, "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz")
            cand = sorted(glob.glob(patt))
            if not cand:
                raise FileNotFoundError("no MC GCD found")
            gcd_file = cand[-1]
        else:
            # ←—— This is the only change you needed
            gcd_file = find_data_gcd(i3_path)

    gf = dataio.I3File(gcd_file)
    cali = None
    while gf.more():
        fr = gf.pop_frame()
        if fr.Stop == I3Frame.Calibration and "I3Calibration" in fr:
            cali = fr["I3Calibration"]
            break
    gf.close()
    if cali is None:
        raise RuntimeError(f"no I3Calibration found in GCD file: {gcd_file}")
    return cali, gcd_file

from analysis_tools.utils.file_search import find_files_for_runs


def plot_event_pulses(
    df: pd.DataFrame,
    master_dir: str,
    *,
    run_col: str = "I3EventHeader_Run",
    event_col: str = "I3EventHeader_Event",
    run_id: Optional[int] = None,            # if provided, filter to this run
    event_id: Optional[int] = None,          # if provided, filter to this event (with run)
    pulse_key: str = "InIcePulses",
    string: Optional[int] = None,
    dom_range: Optional[Tuple[int,int]] = None,   # if None: auto from hits
    time_window: float = 500.0,
    time_anchor: str = "min",        # "min" | "quantile" | "peak"  (you already have this)
    anchor_quantile: float = 0.05,   # used when time_anchor="quantile"
    prepad_ns: float = 200.0, 
    auto_recenter: bool = True,
    bins: Union[int, Tuple[int,int]] = 20,        # int = time bins; DOM bins auto
    norm: Literal["log","linear"] = "log",
    cmap: str = "viridis",
    cmin: float = 1.0,
    figsize: Tuple[float,float] = (8,4),
    xlabel: str = "Time [ns]",
    ylabel: str = "DOM",
    title: Optional[str] = None,
    colorbar_label: str = "Charge [p.e.]",
    show: bool = True,
    save_dir: Optional[str] = None,
    gcd_mode: Literal["mc","data"] = "mc",
    gcd_file: Optional[str] = None,
):
    """
    2D time vs DOM pulse map for events specified in a DataFrame.
    Automatically finds I3 files (via find_files_for_runs), loads GCD, and plots.

    If (run_id, event_id) are provided → plot only that event; otherwise iterate all rows.
    """

    # 1) Attach i3_path via your finder (must provide an 'i3_path' column)
    df_paths = find_files_for_runs(df, master_dir, run_col=run_col, return_as="dataframe")
    if "i3_path" not in df_paths.columns:
        raise RuntimeError("find_files_for_runs must return a DataFrame with an 'i3_path' column")

    # Optional filter to one event
    if run_id is not None and event_id is not None:
        rows = df_paths[(df_paths[run_col] == run_id) & (df_paths[event_col] == event_id)]
    else:
        rows = df_paths

    # 2) Iterate events
    for _, row in rows[[run_col, event_col, "i3_path"]].iterrows():
        i3_path = row["i3_path"]
        rid     = int(row[run_col])
        eid     = int(row[event_col])

        if not i3_path or not os.path.exists(i3_path):
            print(f"[skip] no file found for run={rid}")
            continue

        # ----- GCD calibration -----
        gcd_to_use = gcd_file
        if gcd_to_use is None:
            if gcd_mode == "mc":
                base = "/cvmfs/icecube.opensciencegrid.org/data/GCD/"
                patt = os.path.join(base, "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz")
                found = sorted(glob.glob(patt))
                if not found:
                    print("[skip] no MC GCD found")
                    continue
                gcd_to_use = found[-1]
            else:
                gcd_to_use = find_data_gcd(i3_path)

        gf = dataio.I3File(gcd_to_use)
        cali = None
        while gf.more():
            fr = gf.pop_frame()
            if fr.Stop == I3Frame.Calibration and "I3Calibration" in fr:
                cali = fr["I3Calibration"]
                break
        gf.close()
        if cali is None:
            print(f"[skip] no I3Calibration in GCD for run={rid}")
            continue

        # ----- Load event frame -----
        f = dataio.I3File(i3_path)
        phys = None
        while f.more():
            fr = f.pop_physics()
            hdr = fr["I3EventHeader"]
            if hdr.event_id == eid and hdr.run_id == rid:
                fr.Put("I3Calibration", cali)
                phys = fr
                break
        f.close()
        if phys is None:
            print(f"[skip] no matching physics frame for run={rid}, event={eid}")
            continue

        # ----- helper: plot one string -----
        def _plot_for_string(
            s: int,
            dr: Optional[Tuple[int,int]],
        ):
            pm = dataclasses.I3RecoPulseSeriesMap.from_frame(phys, pulse_key)
            ts, ds, ws = [], [], []
        
            # Iterate DOMs: auto all (0..59) or user range
            dom_lo = 0 if dr is None else dr[0]
            dom_hi = 60 if dr is None else dr[1]
        
            for dom in range(dom_lo, dom_hi):
                om = OMKey(s, dom)
                for p in pm.get(om, []):
                    if p.flags & dataclasses.I3RecoPulse.PulseFlags.ATWD:
                        ts.append(p.time)
                        ds.append(dom)
                        ws.append(p.charge)
        
            if not ts:
                print(f"→ no pulses for string {s}")
                return
        
            arr_t = np.array(ts); arr_d = np.array(ds); arr_w = np.array(ws)
        
            # Auto DOM range from hits if not provided
            if dr is None:
                uniq = np.unique(arr_d)
                dr0, dr1 = int(uniq.min()), int(uniq.max()) + 1
            else:
                dr0, dr1 = dr
        
            # time bins (DOM bins = one per integer DOM)
            nt = bins if isinstance(bins, int) else int(bins[0])
        
            def make_hist(t0_val: float, cmin_val: float):
                xedges = np.linspace(t0_val, t0_val + time_window, nt + 1)
                yedges = np.arange(dr0, dr1 + 1)
                counts, _, _ = np.histogram2d(arr_t, arr_d, bins=[xedges, yedges], weights=arr_w)
                masked = np.ma.masked_where(counts <= cmin_val, counts)
                return masked, xedges, yedges
        
            def pick_norm(masked: np.ma.MaskedArray):
                if masked.count() == 0:
                    return None
                vals = masked.compressed()
                if vals.size and np.all(vals > 0):
                    vmin = max(vals.min(), 1e-12)
                    vmax = vals.max()
                    return LogNorm(vmin=vmin, vmax=vmax)
                else:
                    return Normalize(vmin=(vals.min() if vals.size else 0.0),
                                     vmax=(vals.max() if vals.size else 1.0))
        
            # --- choose t0 with prepad ---
            t0 = arr_t.min() - prepad_ns
            masked, xedges, yedges = make_hist(t0, cmin)
            norm_obj = pick_norm(masked)
        
            # Fallbacks same as before
            if norm_obj is None:
                t0_q = np.quantile(arr_t, 0.05) - prepad_ns
                masked, xedges, yedges = make_hist(t0_q, cmin)
                norm_obj = pick_norm(masked)
        
            if norm_obj is None and arr_t.ptp() > 0:
                tmin, tmax = arr_t.min(), arr_t.max()
                nb = max(100, min(2000, arr_t.size))
                edges = np.linspace(tmin, tmax, nb + 1)
                hist, _ = np.histogram(arr_t, bins=edges)
                bin_dt = (tmax - tmin) / nb
                k = max(1, int(np.ceil(time_window / bin_dt)))
                cs = np.cumsum(np.r_[0, hist])
                win_sums = cs[k:] - cs[:-k]
                j = int(np.argmax(win_sums)) if win_sums.size else 0
                t0_peak = edges[j] - prepad
                masked, xedges, yedges = make_hist(t0_peak, cmin)
                norm_obj = pick_norm(masked)
        
            if norm_obj is None and cmin > 0:
                masked, xedges, yedges = make_hist(arr_t.min() - prepad, max(0.0, cmin * 0.1))
                norm_obj = pick_norm(masked)
        
            if norm_obj is None:
                print(f"[skip] no visible bins for string {s}")
                return
        
            # Plot
            fig, ax = plt.subplots(figsize=figsize)
            mesh = ax.pcolormesh(xedges, yedges, masked.T, norm=norm_obj, cmap=cmap, shading="auto")
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label(colorbar_label)
        
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title or f"Run {rid}  Event {eid}  String {s}")
            fig.tight_layout()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"run{rid}_evt{eid}_str{s}.png"), bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)

        # ----- dispatch: specific string vs BrightDOMs groups -----
        if string is not None:
            _plot_for_string(string, dom_range)
        else:
            if "BrightDOMs" not in phys:
                print("[skip] no BrightDOMs and string=None")
                continue

            # group by string from BrightDOMs
            str_to_doms: Dict[int, list] = {}
            for om in phys["BrightDOMs"]:
                str_to_doms.setdefault(om.string, []).append(om.om)

            pm = dataclasses.I3RecoPulseSeriesMap.from_frame(phys, pulse_key)
            # For each listed string, widen to ALL hit DOMs (not just those listed)
            for s in str_to_doms:
                hit_doms = [om[0][1] for om in pm if om[0][0] == s]
                if not hit_doms:
                    continue
                dr0, dr1 = min(hit_doms), max(hit_doms) + 1
                _plot_for_string(s, (dr0, dr1))


def plot_dom_pulses(
    df: pd.DataFrame,
    master_dir: str,
    *,
    run_col: str = "I3EventHeader_Run",
    event_col: str = "I3EventHeader_Event",
    run_id: Optional[int] = None,
    event_id: Optional[int] = None,
    pulse_key: str = "InIcePulses",
    string: Optional[int] = None,
    dom: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 4),
    xlabel: str = "Time [ns]",
    ylabel: str = "Charge [p.e.]",
    title: Optional[str] = None,
    show: bool = True,
    save_dir: Optional[str] = None,
    gcd_mode: str = "mc",                 # "mc" or "data"
    gcd_file: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    time_window: Optional[float] = None, 
):
    """
    Make per-DOM pulse plots (charge vs time) for events listed in a DataFrame.
    If time_window is provided (and xlim is not), x-axis will be
    [first_pulse_time, first_pulse_time + time_window].
    """

    # Resolve I3 file paths
    df_paths = find_files_for_runs(df, master_dir, run_col=run_col, return_as="dataframe")
    if "i3_path" not in df_paths.columns:
        raise RuntimeError("find_files_for_runs must return an 'i3_path' column")

    # Select rows to plot
    if run_id is not None and event_id is not None:
        rows = df_paths[(df_paths[run_col] == run_id) & (df_paths[event_col] == event_id)]
    else:
        rows = df_paths

    # Iterate events
    for _, row in rows[[run_col, event_col, "i3_path"]].iterrows():
        i3_path = row["i3_path"]
        rid     = int(row[run_col])
        eid     = int(row[event_col])

        if not i3_path or not os.path.exists(i3_path):
            print(f"[skip] no file found for run={rid}")
            continue

        # --- GCD selection ---
        gcd_to_use = gcd_file
        if gcd_to_use is None:
            if gcd_mode.lower() == "mc":
                base = "/cvmfs/icecube.opensciencegrid.org/data/GCD/"
                patt = os.path.join(base, "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz")
                found = sorted(glob.glob(patt))
                if not found:
                    print("[skip] no MC GCD found")
                    continue
                gcd_to_use = found[-1]
            else:
                gcd_to_use = find_data_gcd(i3_path)

        # Load calibration
        gf = dataio.I3File(gcd_to_use)
        cali = None
        while gf.more():
            fr = gf.pop_frame()
            if fr.Stop == I3Frame.Calibration and "I3Calibration" in fr:
                cali = fr["I3Calibration"]
                break
        gf.close()
        if cali is None:
            print(f"[skip] no I3Calibration in GCD for run={rid}")
            continue

        # Load event & inject calibration
        f = dataio.I3File(i3_path)
        phys = None
        while f.more():
            fr = f.pop_physics()
            hdr = fr["I3EventHeader"]
            if hdr.event_id == eid and hdr.run_id == rid:
                fr.Put("I3Calibration", cali)
                phys = fr
                break
        f.close()
        if phys is None:
            print(f"[skip] no matching physics frame for run={rid}, event={eid}")
            continue

        pm = dataclasses.I3RecoPulseSeriesMap.from_frame(phys, pulse_key)

        # === DOM plotting (your original with time_window support) ===
        def _plot_for_dom(omkey: OMKey):
            pulses = pm.get(omkey, [])
            if not pulses:
                print(f"No pulses on OM({omkey.string},{omkey.om})")
                return

            ts = [p.time   for p in pulses if (p.flags & dataclasses.I3RecoPulse.PulseFlags.ATWD)]
            qs = [p.charge for p in pulses if (p.flags & dataclasses.I3RecoPulse.PulseFlags.ATWD)]
            if not ts:
                print(f"No ATWD pulses on OM({omkey.string},{omkey.om})")
                return

            arr_t = np.array(ts)
            arr_q = np.array(qs)

            # Sort by time
            order = np.argsort(arr_t)
            arr_t = arr_t[order]
            arr_q = arr_q[order]

            # Relative suppression threshold (1% of total charge)
            total_q = float(np.sum(arr_q))
            rel_thr = 0.01 * total_q if total_q > 0 else 0.0

            for i in range(len(arr_t)):
                t_cut = arr_t[i]
                mask  = arr_t >= t_cut
                if np.all(arr_q[mask] < rel_thr):
                    arr_t = arr_t[arr_t < t_cut]
                    arr_q = arr_q[:arr_t.size]
                    break

            if arr_t.size == 0:
                print(f"All pulses suppressed for OM({omkey.string},{omkey.om})")
                return

            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(arr_t, arr_q, linestyle='-', lw=1)

            # x-limits priority: explicit xlim > time_window > full data span
            if xlim is not None:
                ax.set_xlim(*xlim)
            elif time_window is not None and time_window > 0:
                t0 = float(arr_t.min())
                ax.set_xlim(t0, t0 + float(time_window))
            else:
                ax.set_xlim(float(arr_t.min()), float(arr_t.max()))

            if ylim is not None:
                ax.set_ylim(*ylim)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title or f"Pulse Q vs T: Run {rid}  Event {eid}  OM({omkey.string},{omkey.om})")
            fig.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                out = os.path.join(save_dir, f"run{rid}_evt{eid}_str{omkey.string}_dom{omkey.om}.png")
                fig.savefig(out, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

        # Dispatch: specific OM or BrightDOMs list
        if string is not None and dom is not None:
            _plot_for_dom(OMKey(string, dom))
        else:
            if "BrightDOMs" not in phys:
                print("[skip] no 'BrightDOMs' in frame and no (string, dom) provided")
                continue
            for om in phys["BrightDOMs"]:
                _plot_for_dom(OMKey(om.string, om.om))
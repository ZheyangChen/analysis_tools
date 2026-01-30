import os
import re
import glob
from typing import Optional, Union, Tuple, Literal, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from icecube import dataio, dataclasses, icetray
from icecube.icetray import OMKey, I3Frame, I3Units

from icecube import dataio
from icecube.icetray import I3Frame
from icecube.dataclasses import I3Geometry, I3Calibration

def load_geometry_and_calibration(i3_path: str, gcd_mode: str = "mc", gcd_file: Optional[str] = None) -> Tuple[I3Geometry, I3Calibration, str]:
    """
    Returns (geo, cali, gcd_file). If no gcd_file is given, it will auto-detect:
    - For MC: hardcoded path.
    - For Data: uses find_data_gcd(i3_path).
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
            from analysis_tools.utils.file_search import find_gcd_for_i3
            gcd_file = find_gcd_for_i3(i3_path)

    geo = None
    cali = None
    f = dataio.I3File(gcd_file)
    while f.more():
        fr = f.pop_frame()
        if fr.Stop == I3Frame.Calibration and "I3Calibration" in fr:
            cali = fr["I3Calibration"]
        elif fr.Stop == I3Frame.Geometry and "I3Geometry" in fr:
            geo = fr["I3Geometry"]
        if geo and cali:
            break
    f.close()

    if cali is None:
        raise RuntimeError(f"no I3Calibration found in GCD file: {gcd_file}")
    if geo is None:
        raise RuntimeError(f"no I3Geometry found in GCD file: {gcd_file}")

    return geo, cali, gcd_file
    
from analysis_tools.utils.file_search import find_i3_files


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
    time_range: Optional[Tuple[float, float]] = None,  # (t_start, t_end) overrides time_window
    full_time_range: bool = False,  # if True, ignore time_window and use full pulse span
    time_anchor: str = "min",        # "min" | "quantile" | "peak"  (you already have this)
    anchor_quantile: float = 0.05,   # used when time_anchor="quantile"
    prepad_ns: float = 200.0, 
    auto_recenter: bool = True,
    bins: Union[int, Tuple[int,int]] = 20,        # int = time bins; DOM bins auto
    norm: Literal["log","linear"] = "log",
    cmap: str = "viridis",
    cmin: float = 0.5,
    figsize: Tuple[float,float] = (8,4),
    xlabel: str = "Time [ns]",
    ylabel: str = "DOM",
    title: Optional[str] = None,
    colorbar_label: str = "Charge [p.e.]",
    show: bool = True,
    save_dir: Optional[str] = None,
    gcd_mode: Literal["mc","data"] = "mc",
    gcd_file: Optional[str] = None,
    i3_path: Optional[str] = None,
    dom_selection: Literal["bright", "nearest"] = "bright",
    vertex_key: str = "cscdSBU_MonopodFit4_noDC",
    nearest_dom_count: int = 10,
):
    """
    2D time vs DOM pulse map for events specified in a DataFrame.
    Automatically finds I3 files (via find_files_for_runs), loads GCD, and plots.

    If (run_id, event_id) are provided → plot only that event; otherwise iterate all rows.
    Time range selection precedence: time_range > full_time_range > time_window.
    """

    # 1) Attach i3_path via your finder (must provide an 'i3_path' column)
    if i3_path:
    # Direct I3 input mode
        df_paths = pd.DataFrame({
            run_col: [run_id],
            event_col: [event_id],
            "i3_path": [i3_path]
        })
    else:
        # Standard mode: resolve from master_dir
        if run_id is not None:
            df_sub = df[df[run_col] == run_id]
        else:
            df_sub = df
        
        df_paths = find_i3_files(df_sub, master_dir, run_col=run_col, mode=gcd_mode, return_as="dataframe")
        
    if "i3_path" not in df_paths.columns:
        raise RuntimeError("find_files_for_runs must return a DataFrame with an 'i3_path' column")

    # Optional filter to one event
    if run_id is not None and event_id is not None:
        rows = df_paths[(df_paths[run_col] == run_id) & (df_paths[event_col] == event_id)]
    else:
        rows = df_paths

    # 2) Iterate events
    fig, ax = None, None
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
            if gcd_mode.lower() == "mc":
                base = "/cvmfs/icecube.opensciencegrid.org/data/GCD/"
                patt = os.path.join(base, "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz")
                found = sorted(glob.glob(patt))
                if not found:
                    print("[skip] no MC GCD found")
                    continue
                gcd_to_use = found[-1]
            else:
                from analysis_tools.utils.file_search import find_gcd_for_i3
                gcd_to_use = find_gcd_for_i3(i3_path)

        cache_key = gcd_file or i3_path
        if 'gcd_cache' not in locals():
            gcd_cache = {}
        if cache_key not in gcd_cache:
            geo, cali, gcd_to_use = load_geometry_and_calibration(i3_path, gcd_mode=gcd_mode, gcd_file=gcd_file)
            gcd_cache[cache_key] = (geo, cali, gcd_to_use)
        geo, cali, gcd_to_use = gcd_cache[cache_key]
        if geo is None:
            print(f"[skip] no I3Geometry in GCD for run={rid}")
            continue
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
        
            def make_hist(t0_val: float, cmin_val: float, window: float):
                xedges = np.linspace(t0_val, t0_val + window, nt + 1)
                yedges = np.arange(dr0, dr1 + 1)
                counts, _, _ = np.histogram2d(arr_t, arr_d, bins=[xedges, yedges], weights=arr_w)
                masked = np.ma.masked_where(counts <= cmin_val, counts)
                return masked, xedges, yedges
        
            def pick_norm(masked: np.ma.MaskedArray):
                if masked.count() == 0:
                    return None
                vals = masked.compressed()
                vmin = max(vals.min(), 1e-12) if vals.size else 1e-12
                vmax = vals.max() if vals.size else 1.0
            
                if norm == "log":
                    return LogNorm(vmin=vmin, vmax=vmax)
                else:
                    return Normalize(vmin=vmin, vmax=vmax)
        
            # --- choose t0 with prepad ---
            # time range selection: explicit > full span > default window
            if time_range is not None:
                t_start, t_end = float(time_range[0]), float(time_range[1])
                t0 = t_start
                tw = max(0.0, t_end - t_start)
            elif full_time_range:
                t0 = float(arr_t.min())
                tw = float(arr_t.max()) - float(arr_t.min())
            else:
                t0 = float(arr_t.min()) - prepad_ns
                tw = float(time_window)

            masked, xedges, yedges = make_hist(t0, cmin, tw)
            norm_obj = pick_norm(masked)
        
            # Fallbacks same as before
            if norm_obj is None:
                t0_q = np.quantile(arr_t, 0.05) - prepad_ns
                masked, xedges, yedges = make_hist(t0_q, cmin, tw)
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
                t0_peak = edges[j] - prepad_ns
                masked, xedges, yedges = make_hist(t0_peak, cmin, tw)
                norm_obj = pick_norm(masked)
        
            if norm_obj is None and cmin > 0:
                masked, xedges, yedges = make_hist(arr_t.min() - prepad_ns, max(0.0, cmin * 0.1), tw)
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

            
            return fig, ax

        # ----- dispatch: specific string vs BrightDOMs groups -----
        if string is not None:
            result = _plot_for_string(string, dom_range)
            if result is not None:
                fig, ax = result
        else:
            pm = dataclasses.I3RecoPulseSeriesMap.from_frame(phys, pulse_key)
        
            if dom_selection == "bright":
                if "BrightDOMs" not in phys:
                    print("[skip] no BrightDOMs and string=None")
                    continue
        
                str_to_doms: Dict[int, list] = {}
                for om in phys["BrightDOMs"]:
                    str_to_doms.setdefault(om.string, []).append(om.om)
        
                for s in str_to_doms:
                    hit_doms = [om[0][1] for om in pm if om[0][0] == s]
                    if not hit_doms:
                        continue
                    dr0, dr1 = min(hit_doms), max(hit_doms) + 1
                    result = _plot_for_string(s, (dr0, dr1))
                    if result is not None:
                        fig, ax = result
        
            elif dom_selection == "nearest":
                if vertex_key not in phys:
                    print(f"[skip] no vertex '{vertex_key}' in frame")
                    continue
        
                vertex_pos = phys[vertex_key].pos
                x0, y0, z0 = vertex_pos.x, vertex_pos.y, vertex_pos.z
        
                # Build a list of (distance, OMKey)
                dist_om_pairs = []
                for om, pulses in pm.items():
                    if not pulses:
                        continue
                    pos = geo.omgeo[om].position
                    dx, dy, dz = pos.x - x0, pos.y - y0, pos.z - z0
                    dist2 = dx*dx + dy*dy + dz*dz
                    dist_om_pairs.append((dist2, om))
        
                if not dist_om_pairs:
                    print(f"[skip] no pulses to find nearest DOMs")
                    continue
        
                dist_om_pairs.sort()
                selected = dist_om_pairs[:nearest_dom_count]
        
                # Group selected by string
                str_to_doms: Dict[int, list] = {}
                for _, om in selected:
                    str_to_doms.setdefault(om.string, []).append(om.om)
        
                for s, doms in str_to_doms.items():
                    dr0, dr1 = min(doms), max(doms) + 1
                    result = _plot_for_string(s, (dr0, dr1))
                    if result is not None:
                        fig, ax = result
    return fig, ax

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
    gcd_mode: str = "mc",
    gcd_file: Optional[str] = None,
    i3_path: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    time_window: Optional[float] = None, 
    time_range: Optional[Tuple[float, float]] = None,  # (t_start, t_end) overrides time_window
    full_time_range: bool = False,  # if True, ignore time_window and use full pulse span
    dom_selection: Literal["bright", "nearest"] = "bright",
    vertex_key: str = "cscdSBU_MonopodFit4_noDC",
    nearest_dom_count: int = 10,
):
    """
    Make per-DOM pulse plots (charge vs time) for events listed in a DataFrame.
    If time_window is provided (and xlim is not), x-axis will be
    [first_pulse_time, first_pulse_time + time_window].
    Time range selection precedence: time_range > full_time_range > time_window.
    """

    # Resolve I3 file paths
    if i3_path:
        # Direct I3 input mode
        df_paths = pd.DataFrame({
            run_col: [run_id],
            event_col: [event_id],
            "i3_path": [i3_path]
        })
    else:
        # Only use the relevant run subset from df to avoid full scan
        if run_id is not None:
            df_sub = df[df[run_col] == run_id]
        else:
            df_sub = df

        df_paths = find_i3_files(df_sub, master_dir, run_col=run_col, mode=gcd_mode, return_as="dataframe")

        
    if "i3_path" not in df_paths.columns:
        raise RuntimeError("find_files_for_runs must return an 'i3_path' column")

    # Select rows to plot
    if run_id is not None and event_id is not None:
        rows = df_paths[(df_paths[run_col] == run_id) & (df_paths[event_col] == event_id)]
    else:
        rows = df_paths

    # Iterate events
    gcd_cache = {}
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
                from analysis_tools.utils.file_search import find_gcd_for_i3
                gcd_to_use = find_gcd_for_i3(i3_path)

        # Load calibration
        cache_key = gcd_file or i3_path
        if cache_key not in gcd_cache:
            geo, cali, used_gcd = load_geometry_and_calibration(i3_path, gcd_mode=gcd_mode, gcd_file=gcd_file)
            gcd_cache[cache_key] = (geo, cali, used_gcd)
        geo, cali, gcd_to_use = gcd_cache[cache_key]

        if geo is None:
            print(f"[skip] no I3Geometry in GCD for run={rid}")
            continue
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
            pm = dataclasses.I3RecoPulseSeriesMap.from_frame(phys, pulse_key)
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
            
            # x-limits priority: explicit xlim > time_range > full_time_range > time_window > full data span
            if xlim is not None:
                ax.set_xlim(*xlim)
            elif time_range is not None:
                t_start, t_end = float(time_range[0]), float(time_range[1])
                ax.set_xlim(t_start, t_end)
            elif full_time_range:
                ax.set_xlim(float(arr_t.min()), float(arr_t.max()))
            elif time_window is not None and time_window > 0:
                t0 = float(arr_t.min())
                ax.set_xlim(t0, t0 + float(time_window))
            else:
                duration = float(arr_t.max()) - float(arr_t.min())
                ax.set_xlim(float(arr_t.min()), float(arr_t.min()) + min(duration, 600.0))

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


        # Dispatch: specific OM or BrightDOMs list
        if string is not None and dom is not None:
            _plot_for_dom(OMKey(string, dom))
        
        elif dom_selection == "bright":
            if "BrightDOMs" not in phys:
                print("[skip] no 'BrightDOMs' in frame and no (string, dom) provided")
                continue
            for om in phys["BrightDOMs"]:
                _plot_for_dom(OMKey(om.string, om.om))
        
        elif dom_selection == "nearest":
            if vertex_key not in phys:
                print(f"[skip] no vertex '{vertex_key}' in frame")
                continue

            vertex_pos = phys[vertex_key].pos
            x0, y0, z0 = vertex_pos.x, vertex_pos.y, vertex_pos.z

            print(f"🔍 Run {rid} Event {eid} using vertex '{vertex_key}':")
            print(f"    x = {x0:.2f}, y = {y0:.2f}, z = {z0:.2f}")

            pm = dataclasses.I3RecoPulseSeriesMap.from_frame(phys, pulse_key)
            dist_om_pairs = []
            for om, pulses in pm.items():
                if not pulses:
                    continue
                pos = geo.omgeo[om].position
                dx, dy, dz = pos.x - x0, pos.y - y0, pos.z - z0
                dist2 = dx * dx + dy * dy + dz * dz
                dist_om_pairs.append((dist2, om))

            if not dist_om_pairs:
                print(f"[skip] no pulses to find nearest DOMs")
                continue

            dist_om_pairs.sort()
            selected = dist_om_pairs[:nearest_dom_count]

            print(f"     Nearest {nearest_dom_count} DOMs with pulses:")
            for dist2, om in selected:
                dist = np.sqrt(dist2)
                print(f"      OM({om.string:2d},{om.om:02d})  →  {dist:.1f} m")

            for _, om in selected:
                _plot_for_dom(om)

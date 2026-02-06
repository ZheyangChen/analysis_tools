from typing import Optional, Sequence

import glob
import os
def overlay_vertex_on_dom_plot(
    *,
    ax,
    i3_path: str,
    run_id: int,
    event_id: int,
    vertex_keys: Sequence[str],
    string: int,
    dom_range,
    pulse_key: str = "InIcePulses",
    gcd_mode: str = "data",
    gcd_file: str = None,
    color: str = "red",
    marker: str = "x",
    ms: int = 10,
    ice_index: float = 1.4,
):
    """
    Overlay reconstructed vertex positions on an existing DOM pulse plot.

    Assumes:
      - ax comes from plot_event_pulses
      - y-axis is DOM index (integer bins)
      - x-axis is time [ns]

    Prints:
      - Full vertex (x, y, z, t)
      - Distance to nearest 3 DOMs with pulses
    """

    from icecube import dataio, dataclasses
    from icecube.icetray import OMKey
    import numpy as np
    from analysis_tools.plotters.waveform_plot import load_geometry_and_calibration

    # -------------------------
    # Load GCD (geometry only)
    # -------------------------
    geo = None
    # Load GCD calibration
    if gcd_file is None:
        if gcd_mode.lower() == "mc":
            base = "/cvmfs/icecube.opensciencegrid.org/data/GCD/"
            patt = os.path.join(base, "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz")
            found = sorted(glob.glob(patt))
            if not found:
                raise RuntimeError("No GCD file found for MC mode")
            gcd_file = found[-1]
        else:
            from analysis_tools.utils.file_search import find_gcd_for_i3
            gcd_file = find_gcd_for_i3(i3_path)
    
    geo, cali, _ = load_geometry_and_calibration(i3_path, gcd_mode=gcd_mode, gcd_file=gcd_file)

    # -------------------------
    # Load physics frame
    # -------------------------
    f = dataio.I3File(i3_path)
    phys = None
    while f.more():
        fr = f.pop_physics()
        hdr = fr["I3EventHeader"]
        if hdr.run_id == run_id and hdr.event_id == event_id:
            fr.Put("I3Calibration", cali)  # ← REQUIRED for pulses
            phys = fr
            break
    f.close()
    
    if phys is None:
        raise RuntimeError(f"No matching physics frame for run={run_id}, event={event_id}")

    # -------------------------
    # Pulse map (for distances)
    # -------------------------
    pm = dataclasses.I3RecoPulseSeriesMap.from_frame(phys, pulse_key)

    dom_lo, dom_hi = dom_range

    # -------------------------
    # Loop over vertices
    # -------------------------
    for key in vertex_keys:
        if key not in phys:
            print(f"[skip] vertex '{key}' not in frame")
            continue

        v = phys[key]
        if not hasattr(v, "pos") or not hasattr(v, "time"):
            print(f"[skip] vertex '{key}' has no pos/time")
            continue

        x0, y0, z0 = v.pos.x, v.pos.y, v.pos.z
        e0 = v.energy
        t0 = v.time

        print(f"\n🔴 Vertex '{key}':")
        print(f"    energy = {e0:.2f} GeV")
        print(f"    x = {x0:.2f} m")
        print(f"    y = {y0:.2f} m")
        print(f"    z = {z0:.2f} m")
        print(f"    t = {t0:.2f} ns")

        # -------------------------
        # Convert z → DOM index
        # -------------------------
        dom_z_map = []
        for dom in range(dom_lo, dom_hi):
            om = OMKey(string, dom)
            if om not in geo.omgeo:
                continue
            dom_z = geo.omgeo[om].position.z
            dom_z_map.append((abs(dom_z - z0), dom, dom_z))

        if not dom_z_map:
            print("    [warn] no DOMs on this string in range")
            continue

        dom_z_map.sort()
        dom_vertex = dom_z_map[0][1]

        # -------------------------
        # Overlay marker
        # -------------------------
        ax.scatter(
            t0,
            dom_vertex,
            color=color,
            marker=marker,
            s=ms**2,
            zorder=10,
            label=key,
        )

        # -------------------------
        # Nearest DOMs with pulses
        # -------------------------
        c_ice = 0.2998/ice_index  # m/ns
        
        dist_om_pairs = []
        for om, pulses in pm.items():
            if not pulses:
                continue
            if string is not None and om.string != string:
                continue  # only consider DOMs on the given string
            pos = geo.omgeo[om].position
            dx = pos.x - x0
            dy = pos.y - y0
            dz = pos.z - z0
            dist2 = dx * dx + dy * dy + dz * dz
            dist_om_pairs.append((dist2, om, pos))
        
        if dist_om_pairs:
            dist_om_pairs.sort()
            print("    Nearest DOMs with pulses:")
            for d2, om, pos in dist_om_pairs[:5]:
                d = np.sqrt(d2)
                t_flight = d / c_ice  # ns
        
                if t0 is not None:
                    t_expected = t0 + t_flight
                    time_str = (
                        f",  flight = {t_flight:.2f} ns"
                        f",  expected arrival = {t_expected:.2f} ns"
                    )
                else:
                    time_str = f",  flight = {t_flight:.2f} ns"

                print(
                    f"      OM({om.string},{om.om}): "
                    f"{d:.2f} m  at (x={pos.x:.1f}, y={pos.y:.1f}, z={pos.z:.1f})"
                    f"{time_str}"
                )
        else:
            print("    [warn] no pulses for nearest-DOM search")

    ax.legend(loc="best")

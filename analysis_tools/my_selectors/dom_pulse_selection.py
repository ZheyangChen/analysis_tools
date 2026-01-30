from typing import Dict, Optional, Tuple, Literal, List, Any

import numpy as np

from icecube import dataclasses, dataio
from analysis_tools.plotters.waveform_plot import load_geometry_and_calibration
from icecube.icetray import I3Frame, OMKey


def find_doms_in_time_range(
    phys: I3Frame,
    *,
    pulse_key: str = "InIcePulses",
    time_range: Tuple[float, float],
    atwd_only: bool = True,
    pulse_mode: Optional[Literal["atwd", "fadc", "both"]] = None,
    lc_mode: Literal["any", "lc", "nolc"] = "any",
    include_pulses: bool = False,
    min_pulse_count: Optional[int] = None,
    max_pulse_count: Optional[int] = None,
    min_total_charge: Optional[float] = None,
    max_total_charge: Optional[float] = None,
    min_max_pulse_charge: Optional[float] = None,
    max_max_pulse_charge: Optional[float] = None,
) -> Dict[OMKey, Dict[str, Any]]:
    """
    Find DOMs with pulses in a custom time range.

    Selection can be done by pulse count, total charge, or max pulse charge
    within the time window. Any provided min/max constraint is applied.
    If pulse_mode is provided, it overrides atwd_only.

    Returns a dict: OMKey -> {"count": int, "total_charge": float, "max_charge": float}
    If include_pulses is True, adds "pulses": [{"time": float, "charge": float, "width": Optional[float]}, ...]
    """

    t0, t1 = float(time_range[0]), float(time_range[1])
    if t1 < t0:
        t0, t1 = t1, t0

    if pulse_mode is None:
        pulse_mode = "atwd" if atwd_only else "both"

    pm = dataclasses.I3RecoPulseSeriesMap.from_frame(phys, pulse_key)
    out: Dict[OMKey, Dict[str, float]] = {}

    for om, pulses in pm.items():
        if not pulses:
            continue

        ts: List[float] = []
        qs: List[float] = []
        widths: List[Optional[float]] = []
        for p in pulses:
            flags = p.flags
            if pulse_mode == "atwd" and not (flags & dataclasses.I3RecoPulse.PulseFlags.ATWD):
                continue
            if pulse_mode == "fadc" and not (flags & dataclasses.I3RecoPulse.PulseFlags.FADC):
                continue
            if pulse_mode == "both" and not (
                (flags & dataclasses.I3RecoPulse.PulseFlags.ATWD)
                or (flags & dataclasses.I3RecoPulse.PulseFlags.FADC)
            ):
                continue
            if lc_mode == "lc" and not (flags & dataclasses.I3RecoPulse.PulseFlags.LC):
                continue
            if lc_mode == "nolc" and (flags & dataclasses.I3RecoPulse.PulseFlags.LC):
                continue
            if t0 <= p.time <= t1:
                ts.append(p.time)
                qs.append(p.charge)
                widths.append(getattr(p, "width", None))

        if not ts:
            continue

        count = len(ts)
        total_charge = float(np.sum(qs))
        max_charge = float(np.max(qs)) if qs else 0.0

        if min_pulse_count is not None and count < min_pulse_count:
            continue
        if max_pulse_count is not None and count > max_pulse_count:
            continue
        if min_total_charge is not None and total_charge < min_total_charge:
            continue
        if max_total_charge is not None and total_charge > max_total_charge:
            continue
        if min_max_pulse_charge is not None and max_charge < min_max_pulse_charge:
            continue
        if max_max_pulse_charge is not None and max_charge > max_max_pulse_charge:
            continue

        entry: Dict[str, Any] = {
            "count": count,
            "total_charge": total_charge,
            "max_charge": max_charge,
        }
        if include_pulses:
            entry["pulses"] = [
                {"time": float(t), "charge": float(q), "width": w}
                for t, q, w in zip(ts, qs, widths)
            ]

        out[om] = entry

    return out


def find_doms_in_time_range_from_i3(
    i3_path: str,
    *,
    run_id: int,
    event_id: int,
    pulse_key: str = "InIcePulses",
    time_range: Tuple[float, float],
    atwd_only: bool = True,
    pulse_mode: Optional[Literal["atwd", "fadc", "both"]] = None,
    lc_mode: Literal["any", "lc", "nolc"] = "any",
    include_pulses: bool = False,
    gcd_mode: Literal["mc", "data"] = "data",
    gcd_file: Optional[str] = None,
    min_pulse_count: Optional[int] = None,
    max_pulse_count: Optional[int] = None,
    min_total_charge: Optional[float] = None,
    max_total_charge: Optional[float] = None,
    min_max_pulse_charge: Optional[float] = None,
    max_max_pulse_charge: Optional[float] = None,
) -> Dict[OMKey, Dict[str, float]]:
    """
    Load an I3 file, find the target physics frame, and select DOMs in a time range.

    Note: GCD is not required for pulse-only selection, so this loader only uses
    i3_path + (run_id, event_id) to extract the physics frame.
    """

    # Load calibration from GCD to satisfy I3RecoPulseSeriesMap.from_frame
    _, cali, _ = load_geometry_and_calibration(i3_path, gcd_mode=gcd_mode, gcd_file=gcd_file)

    f = dataio.I3File(i3_path)
    phys = None
    while f.more():
        fr = f.pop_physics()
        hdr = fr["I3EventHeader"]
        if hdr.event_id == event_id and hdr.run_id == run_id:
            fr.Put("I3Calibration", cali)
            phys = fr
            break
    f.close()

    if phys is None:
        raise RuntimeError(
            f"no matching physics frame for run={run_id}, event={event_id} in {i3_path}"
        )

    return find_doms_in_time_range(
        phys,
        pulse_key=pulse_key,
        time_range=time_range,
        atwd_only=atwd_only,
        pulse_mode=pulse_mode,
        lc_mode=lc_mode,
        include_pulses=include_pulses,
        min_pulse_count=min_pulse_count,
        max_pulse_count=max_pulse_count,
        min_total_charge=min_total_charge,
        max_total_charge=max_total_charge,
        min_max_pulse_charge=min_max_pulse_charge,
        max_max_pulse_charge=max_max_pulse_charge,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find DOMs with pulses in a time range.")
    parser.add_argument("--i3-path", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--t1", type=float, required=True)
    parser.add_argument("--pulse-key", default="InIcePulses")
    parser.add_argument("--pulse-mode", choices=["atwd", "fadc", "both"])
    parser.add_argument("--lc-mode", choices=["any", "lc", "nolc"], default="any")
    parser.add_argument("--show-pulses", action="store_true", default=False)
    parser.add_argument("--gcd-mode", choices=["mc", "data"], default="data")
    parser.add_argument("--gcd-file")
    parser.add_argument("--atwd-only", action="store_true", default=True)
    parser.add_argument("--no-atwd-only", dest="atwd_only", action="store_false")
    parser.add_argument("--min-pulse-count", type=int)
    parser.add_argument("--max-pulse-count", type=int)
    parser.add_argument("--min-total-charge", type=float)
    parser.add_argument("--max-total-charge", type=float)
    parser.add_argument("--min-max-pulse-charge", type=float)
    parser.add_argument("--max-max-pulse-charge", type=float)
    args = parser.parse_args()

    results = find_doms_in_time_range_from_i3(
        args.i3_path,
        run_id=args.run_id,
        event_id=args.event_id,
        pulse_key=args.pulse_key,
        time_range=(args.t0, args.t1),
        atwd_only=args.atwd_only,
        pulse_mode=args.pulse_mode,
        lc_mode=args.lc_mode,
        include_pulses=args.show_pulses,
        gcd_mode=args.gcd_mode,
        gcd_file=args.gcd_file,
        min_pulse_count=args.min_pulse_count,
        max_pulse_count=args.max_pulse_count,
        min_total_charge=args.min_total_charge,
        max_total_charge=args.max_total_charge,
        min_max_pulse_charge=args.min_max_pulse_charge,
        max_max_pulse_charge=args.max_max_pulse_charge,
    )

    if not results:
        print("No DOMs matched.")
    else:
        for om in sorted(results.keys(), key=lambda k: (k.string, k.om)):
            stats = results[om]
            print(
                f"OM({om.string},{om.om}) count={stats['count']} "
                f"total_charge={stats['total_charge']:.3f} max_charge={stats['max_charge']:.3f}"
            )
            if args.show_pulses and "pulses" in stats:
                for p in stats["pulses"]:
                    width = p["width"]
                    width_str = f"{width:.3f}" if isinstance(width, (int, float)) else "None"
                    print(
                        f"  t={p['time']:.3f} q={p['charge']:.3f} width={width_str}"
                    )

from typing import Dict, Optional, Tuple

import numpy as np

from icecube import dataclasses, dataio
from icecube.icetray import I3Frame, OMKey


def find_doms_in_time_range(
    phys: I3Frame,
    *,
    pulse_key: str = "InIcePulses",
    time_range: Tuple[float, float],
    atwd_only: bool = True,
    min_pulse_count: Optional[int] = None,
    max_pulse_count: Optional[int] = None,
    min_total_charge: Optional[float] = None,
    max_total_charge: Optional[float] = None,
    min_max_pulse_charge: Optional[float] = None,
    max_max_pulse_charge: Optional[float] = None,
) -> Dict[OMKey, Dict[str, float]]:
    """
    Find DOMs with pulses in a custom time range.

    Selection can be done by pulse count, total charge, or max pulse charge
    within the time window. Any provided min/max constraint is applied.

    Returns a dict: OMKey -> {"count": int, "total_charge": float, "max_charge": float}
    """

    t0, t1 = float(time_range[0]), float(time_range[1])
    if t1 < t0:
        t0, t1 = t1, t0

    pm = dataclasses.I3RecoPulseSeriesMap.from_frame(phys, pulse_key)
    out: Dict[OMKey, Dict[str, float]] = {}

    for om, pulses in pm.items():
        if not pulses:
            continue

        ts = []
        qs = []
        for p in pulses:
            if atwd_only and not (p.flags & dataclasses.I3RecoPulse.PulseFlags.ATWD):
                continue
            if t0 <= p.time <= t1:
                ts.append(p.time)
                qs.append(p.charge)

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

        out[om] = {
            "count": count,
            "total_charge": total_charge,
            "max_charge": max_charge,
        }

    return out


def find_doms_in_time_range_from_i3(
    i3_path: str,
    *,
    run_id: int,
    event_id: int,
    pulse_key: str = "InIcePulses",
    time_range: Tuple[float, float],
    atwd_only: bool = True,
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

    f = dataio.I3File(i3_path)
    phys = None
    while f.more():
        fr = f.pop_physics()
        hdr = fr["I3EventHeader"]
        if hdr.event_id == event_id and hdr.run_id == run_id:
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

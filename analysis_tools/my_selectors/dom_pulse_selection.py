from typing import Dict, Optional, Tuple

import numpy as np

from icecube import dataclasses
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

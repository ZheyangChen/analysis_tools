try:
    from .dom_pulse_selection import find_doms_in_time_range, find_doms_in_time_range_from_i3
except ModuleNotFoundError:
    # IceCube not available; allow package import in non-IceCube envs.
    pass

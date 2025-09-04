from typing import Iterable, Dict, Any, Optional
import pandas as pd
from analysis_tools.utils.file_search import find_files_for_runs
from analysis_tools.plotters.waveform_plot import plot_event_pulses, plot_dom_pulses

def evaluate_events_with_plots(
    df: pd.DataFrame,
    master_dir: str,
    run_col: str = "run",
    event_col: str = "event",
    modules: Iterable[str] = ("event_pulses", "dom_pulses"),
    common_kwargs: Optional[Dict[str, Any]] = None,
    event_kwargs: Optional[Dict[str, Any]] = None,
    dom_kwargs: Optional[Dict[str, Any]] = None,
):
    common_kwargs = common_kwargs or {}
    event_kwargs = {**common_kwargs, **(event_kwargs or {})}
    dom_kwargs   = {**common_kwargs, **(dom_kwargs or {})}

    for _, row in df.iterrows():
        path = find_file_for_run(row[run_col], master_dir)
        if not path:
            print(f"[WARN] No file for run={row[run_col]}")
            continue
        if "event_pulses" in modules:
            plot_event_pulses(path, int(row[event_col]), int(row[run_col]), **event_kwargs)
        if "dom_pulses" in modules:
            plot_dom_pulses(path, int(row[event_col]), int(row[run_col]), **dom_kwargs)
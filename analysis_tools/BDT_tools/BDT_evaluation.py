import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import pandas as pd

def find_best_two_bdt_thresholds(
    df: pd.DataFrame,
    score1_col: str,
    score2_col: str,
    label_col: str,
    signal_label,
    weight_col: str = None,
    purity_target: float = 0.90,
    n_steps: int = 101
):
    """
    Find the best (t1, t2) thresholds on two BDT scores so that:
      - purity ≥ purity_target
      - ντ retention is maximized

    Parameters
    ----------
    df            : DataFrame with your events
    score1_col    : first BDT’s score column (e.g. τ vs νe/NC)
    score2_col    : second BDT’s score column (e.g. τ vs νμ/CC)
    label_col     : column holding the true class label
    signal_label  : value in label_col that identifies ντ
    weight_col    : (optional) per‐event weight column; if None, all weights=1
    purity_target : minimum acceptable ντ purity (default 0.90)
    n_steps       : how finely to scan each threshold in [0,1] (default 101 → 0.00,0.01,…,1.00)

    Returns
    -------
    dict with keys:
      t1, t2        : the chosen thresholds
      purity        : achieved ντ/(total selected)
      retention     : ντ kept / ντ total
      n_selected    : how many weighted events passed both cuts
    """
    # extract arrays
    s1 = df[score1_col].values
    s2 = df[score2_col].values
    y  = (df[label_col] == signal_label).values.astype(bool)
    w  = df[weight_col].values if weight_col else np.ones_like(s1, dtype=float)

    tot_sig = w[y].sum()

    best = {"t1": 0.0, "t2": 0.0, "purity": 0.0, "retention": 0.0, "n_selected": 0.0}
    thr1_vals = np.linspace(0.0, 1.0, n_steps)
    thr2_vals = np.linspace(0.0, 1.0, n_steps)

    for t1 in thr1_vals:
        mask1 = s1 >= t1
        # skip if no signal would survive first cut
        if w[y & mask1].sum() == 0:
            continue
        for t2 in thr2_vals:
            sel = mask1 & (s2 >= t2)
            w_sel = w[sel]
            if w_sel.sum() == 0:
                continue

            # compute purity and retention
            sig_sel = w_sel[y[sel]].sum()
            purity  = sig_sel / w_sel.sum()
            if purity < purity_target:
                continue

            retention = sig_sel / tot_sig
            if retention > best["retention"]:
                best.update({
                    "t1": t1,
                    "t2": t2,
                    "purity": purity,
                    "retention": retention,
                    "n_selected": w_sel.sum()
                })

    return best



def plot_bdt_threshold_scan(
    df: pd.DataFrame,
    score1_col: str,
    score2_col: str,
    label_col: str,
    signal_label,
    weight_col: Optional[str] = None,
    purity_target: float = 0.90,
    n_steps: int = 101,
    # ── New customization args ──
    title: Optional[str]       = None,
    xlabel: Optional[str]      = None,
    ylabel: Optional[str]      = None,
    xlim: Optional[Tuple[float,float]] = None,
    ylim: Optional[Tuple[float,float]] = None,
    cmap: str                  = 'viridis',
    best_marker_kwargs: Optional[Dict] = None,
    show: bool                 = True,
    save_path: Optional[str]   = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Scan two BDT-score thresholds (t1, t2) to maximize signal retention
    subject to purity >= purity_target, and plot a 2D heatmap.

    Returns: (best_t1, best_t2, retention_masked)

    Customizable plotting via title, labels, limits, colormap, etc.
    """
    # 1) Extract arrays
    s1 = df[score1_col].values
    s2 = df[score2_col].values
    is_sig = (df[label_col] == signal_label).values
    w = df[weight_col].values if weight_col else np.ones_like(s1, float)
    total_sig = w[is_sig].sum()

    # 2) Build threshold grids
    t1_vals = np.linspace(0, 1, n_steps)
    t2_vals = np.linspace(0, 1, n_steps)

    # 3) Allocate purity & retention arrays
    purity_arr    = np.zeros((n_steps, n_steps), float)
    retention_arr = np.zeros_like(purity_arr)

    # 4) Compute purity & retention for each (t1,t2)
    for i, t1 in enumerate(t1_vals):
        mask1 = s1 >= t1
        for j, t2 in enumerate(t2_vals):
            sel = mask1 & (s2 >= t2)
            w_sel = w[sel]
            if w_sel.sum() == 0:
                continue
            sig_sel = w[sel & is_sig].sum()
            purity_arr[i,j]    = sig_sel / w_sel.sum()
            retention_arr[i,j] = sig_sel / total_sig

    # 5) Mask out points below purity target
    retention_masked = np.where(purity_arr >= purity_target, retention_arr, np.nan)

    # 6) Find the best thresholds
    idx_flat = np.nanargmax(retention_masked)
    i_best, j_best = np.unravel_index(idx_flat, retention_masked.shape)
    best_t1 = t1_vals[i_best]
    best_t2 = t2_vals[j_best]
    best_ret = retention_masked[i_best, j_best]

    # ── Plotting ──
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(
        retention_masked.T,
        origin='lower',
        aspect='auto',
        extent=[t1_vals[0], t1_vals[-1], t2_vals[0], t2_vals[-1]],
        cmap=cmap
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Signal retention")

    # Best‐point marker
    bmkw = dict(color='red', marker='o', s=80, edgecolor='white', label=f"best ({best_ret:.2f})")
    if best_marker_kwargs:
        bmkw.update(best_marker_kwargs)
    ax.scatter(best_t1, best_t2, **bmkw)

    # Axis labels & title
    ax.set_xlabel(xlabel   or f"{score1_col} threshold")
    ax.set_ylabel(ylabel   or f"{score2_col} threshold")
    ax.set_title(title     or f"Retention (purity ≥ {purity_target:.2f})")

    # Axis limits
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return best_t1, best_t2, retention_masked
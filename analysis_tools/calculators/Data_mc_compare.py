import numpy as np
import pandas as pd
from scipy.stats import norm,poisson
from typing import Optional
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chi2 as chi2_dist

def compare_data_mc_rate_only(
    data_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    weight_col: str = "weight",
    data_weight_col: Optional[str] = None,
    variable_filter: Optional[str] = None,
    distribution = 'poisson',
):
    """
    Perform overall rate comparison between Data and MC.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data events (each row = one event).
    mc_df : pd.DataFrame
        MC events (each row = one event, with weights).
    weight_col : str
        Column name for MC event weights.
    data_weight_col : str or None
        If provided, this column in data_df will be used as weights.
    variable_filter : str, optional
        Optional query filter applied to both data and MC.

    Returns
    -------
    dict
        Dictionary of comparison results.
    """

    # Apply filter
    if variable_filter:
        data_df = data_df.query(variable_filter)
        mc_df = mc_df.query(variable_filter)

    # ----------------
    # Count/Weight Data
    # ----------------
    if data_weight_col:
        if data_weight_col not in data_df.columns:
            raise ValueError(f"'{data_weight_col}' not found in Data DataFrame.")
        data_weights = data_df[data_weight_col].values
        n_data = data_weights.sum()
        err_data = np.sqrt(np.sum(data_weights**2))
    else:
        n_data = len(data_df)
        err_data = np.sqrt(n_data)

    # ----------------
    # MC Weights
    # ----------------
    if weight_col not in mc_df.columns:
        raise ValueError(f"'{weight_col}' not found in MC DataFrame.")

    weights = mc_df[weight_col].values
    n_mc = weights.sum()
    err_mc = np.sqrt(np.sum(weights**2))

    # ----------------
    # Ratio / Pull / p
    # ----------------
    if distribution == 'poisson':
        ratio = n_data / n_mc if n_mc > 0 else np.nan
        # Two-sided Poisson p-value
        if n_mc > 0:
            if n_data < n_mc:
                p_lower = poisson.cdf(n_data, n_mc)
                p_value = 2 * p_lower
            else:
                p_upper = poisson.sf(n_data - 1, n_mc)
                p_value = 2 * p_upper
            p_value = min(p_value, 1.0)
        else:
            p_value = np.nan
        pull = np.nan
        # ----------------
        # Print Results
        # ----------------
        print(f"--- Data vs MC Rate Comparison ---")
        print(f"Data count:           {n_data:.1f} ± {err_data:.1f}" + (f" (weighted by '{data_weight_col}')" if data_weight_col else ""))
        print(f"MC predicted rate:    {n_mc:.2f} ± {err_mc:.2f} (weighted by '{weight_col}')")
        print(f"Data/MC ratio:        {ratio:.3f}")
        print(f"p-value (Poisson):   {p_value:.3g}")
        print("----------------------------------")

    
    elif distribution == 'gaussian':
    
        ratio = n_data / n_mc if n_mc > 0 else np.nan
        combined_err = np.sqrt(err_data**2 + err_mc**2)
        pull = (n_data - n_mc) / combined_err if combined_err > 0 else np.nan
        p_value = 2 * (1 - norm.cdf(abs(pull))) if not np.isnan(pull) else np.nan

        # ----------------
        # Print Results
        # ----------------
        print(f"--- Data vs MC Rate Comparison ---")
        print(f"Data count:           {n_data:.1f} ± {err_data:.1f}" + (f" (weighted by '{data_weight_col}')" if data_weight_col else ""))
        print(f"MC predicted rate:    {n_mc:.2f} ± {err_mc:.2f} (weighted by '{weight_col}')")
        print(f"Data/MC ratio:        {ratio:.3f}")
        print(f"Pull (D − MC)/σ:      {pull:.2f}")
        print(f"p-value (Gaussian):   {p_value:.3g}")
        print("----------------------------------")

    return {
        "n_data": n_data,
        "err_data": err_data,
        "n_mc": n_mc,
        "err_mc": err_mc,
        "ratio": ratio,
        "pull": pull,
        "p_value": p_value,
    }


def compare_data_mc_full(
    data_df,
    mc_df,
    variable,
    bins,
    data_weight_col,
    mc_weight_col,
    transform_func=None,
    xlabel=None,
    ylabel_top="Events",
    ylabel_ratio="Data/MC",
    title=None,
    xscale=None,
    yscale="log",
    xlim=None,
    ylim_top=None,
    ylim_ratio=(0.5, 1.5),
    color_mc="C0",
    label_data="Data",
    label_mc="MC",
    legend_loc="best",
    show_hist=True,
    show_numbers=True,
):
    """
    Compare Data vs MC distributions both visually and quantitatively.

    Returns a dict of quantitative agreement metrics (χ², KS, Data/MC ratio, etc.).
    You can optionally show the histogram plot and/or print the metrics.

    Parameters
    ----------
    show_hist : bool
        If True, show the histogram + ratio plot.
    show_numbers : bool
        If True, print the numerical metrics to console.
    (Other parameters are identical to before.)
    """

    # ----- Transform data -----
    data_vals = data_df[variable]
    mc_vals = mc_df[variable]
    if transform_func:
        data_vals = transform_func(data_vals)
        mc_vals = transform_func(mc_vals)
    data_w = data_df[data_weight_col]
    mc_w = mc_df[mc_weight_col]

    # ----- Histogram both -----
    data_hist, bin_edges = np.histogram(data_vals, bins=bins, weights=data_w)
    mc_hist, _ = np.histogram(mc_vals, bins=bins, weights=mc_w)
    data_err = np.sqrt(np.histogram(data_vals, bins=bins, weights=data_w**2)[0])
    mc_err = np.sqrt(np.histogram(mc_vals, bins=bins, weights=mc_w**2)[0])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # ----- χ² and KS -----
    mask = (mc_hist > 0) & (data_err > 0)
    chi2_terms = (data_hist[mask] - mc_hist[mask])**2 / (data_err[mask]**2 + mc_err[mask]**2)
    chi2 = np.sum(chi2_terms)
    ndf = np.sum(mask) - 1
    chi2_ndf = chi2 / ndf if ndf > 0 else np.nan
    chi2_pvalue = 1 - chi2_dist.cdf(chi2, ndf) if ndf > 0 else np.nan

    ks_stat, ks_pvalue = ks_2samp(data_vals, mc_vals)

    # ----- Data/MC ratio -----
    ratio = np.full_like(mc_hist, np.nan, dtype=float)
    ratio[mask] = data_hist[mask] / mc_hist[mask]
    ratio_err = np.full_like(mc_hist, np.nan, dtype=float)
    ratio_err[mask] = ratio[mask] * np.sqrt(
        (data_err[mask]/data_hist[mask])**2 + (mc_err[mask]/mc_hist[mask])**2
    )

    mean_ratio = np.nanmean(ratio)
    std_ratio = np.nanstd(ratio)
    sum_data = np.sum(data_hist)
    sum_mc = np.sum(mc_hist)

    # ----- Show numerical results -----
    metrics = {
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2_ndf,
        "chi2_pvalue": chi2_pvalue,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "sum_data": sum_data,
        "sum_mc": sum_mc,
        "mean_ratio": mean_ratio,
        "std_ratio": std_ratio,
    }

    if show_numbers:
        print("\n===== Data/MC Comparison Metrics =====")
        for k, v in metrics.items():
            print(f"{k:>12}: {v:.4f}")
        print("=====================================\n")

    # ----- Plot -----
    if show_hist:
        fig = plt.figure(figsize=(8, 6))
        ax_top = fig.add_axes([0.1, 0.3, 0.85, 0.6])
        ax_ratio = fig.add_axes([0.1, 0.1, 0.85, 0.2], sharex=ax_top)

        # --- top panel ---
        ax_top.hist(
            mc_vals,
            bins=bins,
            weights=mc_w,
            histtype="stepfilled",
            alpha=0.5,
            color=color_mc,
            label=f"{label_mc} (sum={sum_mc:.1f})",
        )
        ax_top.errorbar(
            bin_centers,
            data_hist,
            yerr=data_err,
            fmt="ko",
            label=f"{label_data} (sum={sum_data:.1f})",
            elinewidth=1,
            ms=3,
        )
        ax_top.legend(loc=legend_loc)
        ax_top.set_ylabel(ylabel_top)
        if title:
            ax_top.set_title(title)

        # scales and limits
        if yscale:
            ax_top.set_yscale(yscale)
        if xscale:
            ax_top.set_xscale(xscale)
            ax_ratio.set_xscale(xscale)
        if xlim:
            ax_top.set_xlim(xlim)
            ax_ratio.set_xlim(xlim)
        if ylim_top:
            ax_top.set_ylim(ylim_top)
        if ylim_ratio:
            ax_ratio.set_ylim(ylim_ratio)

        ax_top.tick_params(labelbottom=False)

        # --- ratio panel ---
        ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_err, fmt="ko", elinewidth=1, ms=3)
        ax_ratio.axhline(1, color="black", linestyle="--", linewidth=1)
        ax_ratio.set_xlabel(xlabel if xlabel else variable)
        ax_ratio.set_ylabel(ylabel_ratio)
        ax_ratio.legend([f"χ²/ndf={chi2_ndf:.2f}, p={chi2_pvalue:.3f}"], loc=legend_loc)

        plt.tight_layout()
        plt.show()

    return metrics
# file: analysis_tools/utils/resolution_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Optional, Dict, Any


# ── 1) compute_error_histogram_fit ────────────────────────────────────────────────
def compute_error_histogram_fit(
    df: pd.DataFrame,
    reco_col: str,
    true_col: str,
    weight_col: Optional[str] = None,
    bins: np.ndarray = np.linspace(-1,1,100),
    relative: bool = True,
    fit_window: float = 1.0
) -> Dict[str, Any]:
    """
    Weighted error‐histogram + central‐window Gaussian fit.
    Returns {'centers','hist','popt','pcov'} (popt/pcov may be None).
    """
    # compute per‐event errors
    r = df[reco_col].values
    t = df[true_col].values
    if relative:
        mask = t != 0
        r, t = r[mask], t[mask]
        errors = (r - t) / t
    else:
        errors = np.abs(r - t)

    w = df[weight_col].values if weight_col else np.ones_like(errors, float)

    # raw histogram (no density)
    hist, edges = np.histogram(errors, bins=bins, weights=w, density=False)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    centers = 0.5 * (edges[:-1] + edges[1:])

    # initial guesses
    if total > 0:
        mu0    = np.average(errors, weights=w)
        sigma0 = np.sqrt(np.average((errors-mu0)**2, weights=w))
    else:
        mu0, sigma0 = 0.0, 1.0
    amp0 = 1.0
    p0   = [mu0, sigma0, amp0]

    # mask for central bins
    fit_mask = np.abs(centers - mu0) < fit_window * sigma0

    popt = pcov = None
    if fit_mask.sum() >= 3:
        def gauss(x, mu, sigma, amp):
            return amp * np.exp(-0.5*((x-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
        try:
            popt, pcov = curve_fit(gauss, centers[fit_mask], hist[fit_mask], p0=p0)
        except Exception as e:
            print(f"Warning: skipped error‐hist fit ({e})")

    return {'centers': centers, 'hist': hist, 'popt': popt, 'pcov': pcov}


def compute_reco_resolution(
    df: pd.DataFrame,
    reco_col: str,
    true_col: str,
    weight_col: Optional[str] = None,
    bins: Optional[np.ndarray] = None,
    relative: bool = True
) -> Dict[str, Any]:
    """
    Compute reconstruction resolution δ=(reco−true)/true (if relative)
    or Δ=|reco−true| (if not).

    If bins is None returns global stats:
        { 'mean', 'sigma', 'count' }

    If bins provided returns binned profile:
        {
          'bin_edges', 'bin_centers',
          'mean', 'sigma',
          'raw_counts', 'weighted_counts'
        }
    """
    sub = df[[reco_col, true_col]].dropna()
    if weight_col:
        w = df.loc[sub.index, weight_col].values
    else:
        w = np.ones(len(sub), float)

    r = sub[reco_col].values
    t = sub[true_col].values

    if relative:
        mask = t != 0
        r, t, w = r[mask], t[mask], w[mask]
        delta = (r - t) / t
    else:
        delta = np.abs(r - t)

    # global stats
    if bins is None:
        mean  = np.average(delta, weights=w)
        var   = np.average((delta - mean)**2, weights=w)
        return {'mean': mean, 'sigma': np.sqrt(var), 'count': int(len(delta))}

    # binned profile
    edges = np.asarray(bins)
    idx   = np.digitize(t, edges) - 1
    nbin  = len(edges) - 1

    means        = np.full(nbin, np.nan)
    sigmas       = np.full(nbin, np.nan)
    raw_counts   = np.zeros(nbin, int)
    weighted_cnt = np.zeros(nbin, float)

    for i in range(nbin):
        sel = idx == i
        raw_counts[i] = sel.sum()
        if raw_counts[i] == 0:
            continue
        d_i = delta[sel]
        w_i = w[sel]
        means[i]        = np.average(d_i, weights=w_i)
        sigmas[i]       = np.sqrt(np.average((d_i - means[i])**2, weights=w_i))
        weighted_cnt[i] = w_i.sum()

    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        'bin_edges':       edges,
        'bin_centers':     centers,
        'mean':            means,
        'sigma':           sigmas,
        'raw_counts':      raw_counts,
        'weighted_counts': (weighted_cnt if weight_col else None)
    }


def compute_residual_metrics(
    df: pd.DataFrame,
    reco_col: str,
    true_col: str,
    weight_col: Optional[str] = None,
    relative: bool = True
) -> Dict[str, float]:
    """
    Compute global residual metrics:
      - mean(δ)
      - sigma (RMS)
      - std (weighted std)
      - rms (sqrt(mean(δ^2)))
      - mad (median absolute deviation)
      - iqr (75th-25th percentile width)
    """
    sub = df[[reco_col, true_col]].dropna()
    if weight_col:
        w = df.loc[sub.index, weight_col].values
    else:
        w = np.ones(len(sub), float)

    r = sub[reco_col].values
    t = sub[true_col].values

    if relative:
        mask = t != 0
        r, t, w = r[mask], t[mask], w[mask]
        delta = (r - t) / t
    else:
        delta = np.abs(r - t)

    # weighted mean and std
    mean     = np.average(delta, weights=w)
    var      = np.average((delta - mean)**2, weights=w)
    std      = np.sqrt(var)
    # weighted RMS
    rms      = np.sqrt(np.average(delta**2, weights=w))
    # median and MAD
    median   = np.median(delta)
    mad      = np.median(np.abs(delta - median))
    # IQR
    q75, q25 = np.percentile(delta, [75, 25])
    iqr      = q75 - q25

    # 16th / 84th percentiles for 68% central width
    p16    = np.percentile(delta, 16)
    p84    = np.percentile(delta, 84)
    width68 = p84 - p16


    return {
        'mean': mean,
        'std': std,
        'rms': rms,
        'mad': mad,
        'iqr': iqr,
        'p16':    p16,
        'p84':    p84,
        'width68': width68
    }


def plot_residuals(
    df: pd.DataFrame,
    reco_col: str,
    true_col: str,
    weight_col: Optional[str] = None,
    bins: np.ndarray = np.linspace(-1, 1, 100),
    relative: bool = True,
    show: bool = True,
    save_path: Optional[str] = None
) -> plt.Axes:
    """
    Plot histogram of residuals δ or Δ and annotate global metrics.
    """
    # compute errors
    sub = df[[reco_col, true_col]].dropna()
    if weight_col:
        w = df.loc[sub.index, weight_col].values
    else:
        w = np.ones(len(sub), float)

    r = sub[reco_col].values
    t = sub[true_col].values
    if relative:
        mask = t != 0
        r, t, w = r[mask], t[mask], w[mask]
        errors = (r - t) / t
        xlabel = f"( {reco_col} - {true_col} ) / {true_col}"
    else:
        errors = np.abs(r - t)
        xlabel = f"| {reco_col} - {true_col} |"

    # raw histogram + manual normalization
    hist, edges = np.histogram(errors, bins=bins, weights=w, density=False)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    centers = 0.5 * (edges[:-1] + edges[1:])

    # compute metrics
    metrics = compute_residual_metrics(df, reco_col, true_col, weight_col, relative)

    # plot
    fig, ax = plt.subplots()
    ax.bar(centers, hist, width=centers[1] - centers[0], alpha=0.6, label='residuals')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalized counts')
    title = "Residuals"
    title += f"\nmean={metrics['mean']:.3g}, σ={metrics['std']:.3g}, rms={metrics['rms']:.3g}"
    title += f"\nMAD={metrics['mad']:.3g}, IQR={metrics['iqr']:.3g}"
    ax.set_title(title)
    ax.legend(loc='best')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax


def resolution_workflow(
    df: pd.DataFrame,
    reco_col: str,
    true_col: str,
    weight_col: Optional[str] = None,
    bins: Optional[np.ndarray] = None,
    relative: bool = True,
    profile_fit: bool = True,
    residuals: bool = True,
    error_hist: bool = False,
    residual_bins: Optional[np.ndarray] = None,
    error_hist_bins: Optional[np.ndarray] = None,
    min_count_per_bin: int = 0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    One-stop resolution analysis that also prints a summary of all key values.
    Returns a dict with 'profile', 'fit_params', 'fit_cov',
    'residual_metrics', 'error_hist'.
    """
    results: Dict[str, Any] = {}

    # 1) Compute profile
    prof = compute_reco_resolution(df, reco_col, true_col, weight_col, bins, relative)
    results['profile'] = prof

    # 2) Profile plot & (optional) fit
    fit_params = None
    if bins is not None:
        x_all = prof['bin_centers']
        y_all = prof['sigma']
        counts = prof['raw_counts']
        mask = counts >= min_count_per_bin
        x = x_all[mask]
        y = y_all[mask]

        fig, ax = plt.subplots()
        ax.errorbar(x, y, fmt='o', label='σ')
        ax.set_xscale('log')
        ax.set_xlabel(xlabel or true_col)
        ax.set_ylabel(ylabel or (f"σ(Δ/true)" if relative else "σ|Δ|"))
        if title:
            ax.set_title(title)

        if profile_fit and len(x) >= 3:
            p0 = [y.max() * np.sqrt(x.min()), y.min()]
            try:
                popt, pcov = curve_fit(_default_resolution_fit, x, y, p0=p0)
                xf = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
                ax.plot(xf, _default_resolution_fit(xf, *popt), '-',
                        label=f"fit a/√x+b\na={popt[0]:.3g}, b={popt[1]:.3g}")
                results['fit_params'] = popt
                results['fit_cov']    = pcov
                fit_params = popt
            except Exception as e:
                print(f"Warning: profile fit skipped ({e})")

        ax.legend(loc='best')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path.replace('.png','_profile.png'), bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    # 3) Compute and plot residuals + metrics
    residual_metrics = None
    if residuals:
        residual_metrics = compute_residual_metrics(df, reco_col, true_col, weight_col, relative)
        results['residual_metrics'] = residual_metrics
        plot_residuals(
            df, reco_col, true_col, weight_col,
            bins = residual_bins if residual_bins is not None else np.linspace(-1,1,100),
            relative = relative,
            show     = show,
            save_path= (save_path and save_path.replace('.png','_residuals.png'))
        )

    # 4) Optional error‐histogram + fit
    eh_result = None
    if error_hist:
        eh_bins = error_hist_bins if error_hist_bins is not None else (
            bins if bins is not None else np.linspace(-1,1,100)
        )
        eh_result = compute_error_histogram_fit(
            df, reco_col, true_col, weight_col,
            bins = eh_bins, relative = relative
        )
        results['error_hist'] = eh_result

        centers, hist = eh_result['centers'], eh_result['hist']
        fig, ax = plt.subplots()
        ax.bar(centers, hist, width=centers[1]-centers[0], alpha=0.6, label='data')
        if eh_result['popt'] is not None:
            mu, sigma, amp = eh_result['popt']
            xfit = np.linspace(centers.min(), centers.max(), 300)
            gauss = amp * np.exp(-0.5*((xfit-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
            ax.plot(xfit, gauss, 'r-', label=f"σ={sigma:.3g}")
            ax.legend(loc='best')
        ax.set_xlabel('Error ' + ('(Δ/true)' if relative else '|Δ|'))
        ax.set_ylabel('Density')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path.replace('.png','_errhist.png'), bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ──────────────── Print summary ───────────────────
    print("\n==== Resolution Summary ====")

    # 1) Profile fit parameters
    if fit_params is not None:
        print(f"Profile fit (σ = a/√x + b):")
        print(f"  a = {fit_params[0]:.6g}")
        print(f"  b = {fit_params[1]:.6g}")
    else:
        print("Profile fit: (none)")

    # 2) Global residual metrics
    if residual_metrics is not None:
        print("\nResiduals δ = (reco−true)/true:")
        print(f"  mean       = {residual_metrics['mean']:.6g}")
        print(f"  std        = {residual_metrics['std']:.6g}")
        print(f"  RMS        = {residual_metrics['rms']:.6g}")
        print(f"  MAD        = {residual_metrics['mad']:.6g}")
        print(f"  IQR        = {residual_metrics['iqr']:.6g}")
        print(f"  68% width  = {residual_metrics['width68']:.6g} "
              f"(16th={residual_metrics['p16']:.6g}, 84th={residual_metrics['p84']:.6g})")
    else:
        print("Residual metrics: (none)")

    # 3) Error‐histogram Gaussian core fit
    if eh_result is not None:
        if eh_result.get('popt') is not None:
            mu_fit, sigma_fit, _ = eh_result['popt']
            print("\nError‐histogram fit (central Gaussian):")
            print(f"  μ = {mu_fit:.6g}")
            print(f"  σ = {sigma_fit:.6g}")
        else:
            print("\nError‐histogram fit: (none)")
    # else: error_hist was False, so skip entirely

    print("=============================\n")
    return results
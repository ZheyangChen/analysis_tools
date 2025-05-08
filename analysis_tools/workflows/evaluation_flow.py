import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from matplotlib.colors import LogNorm
from typing import Optional, List, Dict

from analysis_tools.my_selectors.apply_selection    import apply_selection
from analysis_tools.BDT_tools.Testset_preparation  import annotate_labels
from analysis_tools.calculators.event_rates         import compute_rate
from analysis_tools.BDT_tools.BDT_evaluation        import plot_bdt_threshold_scan
from analysis_tools.plotters.histogram_plot         import plot_histograms


def evaluation_flow(
    df: pd.DataFrame,
    model1,
    model2,
    features: list,
    weight_col:      str   = "weight",
    precut_criteria: Optional[dict] = None,
    purity_target:   float = 0.90,
    score1_col:      str   = "bdt1_score",
    score2_col:      str   = "bdt2_score",
    output_dir:      str   = "evaluation_output",
    run_diagnostics: bool  = False
) -> dict:
    """
    1) apply_selection + annotate_labels
    2) predict two BDT scores
    3) print per-flavor pre-BDT rates
    4) plot ROC curves for both BDTs
    5) scan & plot 2D threshold heatmap
    6) print per-flavor post-BDT rates
    7) plot score histograms (nue, numu, nutau) with cut-line legends
    8) plot energy histograms after cuts
    9) optionally run advanced diagnostics
    10) compute purity & efficiency, save summary
    """
    os.makedirs(output_dir, exist_ok=True)
    dfw = df.copy()

    # 1) Pre-cuts + labeling
    if precut_criteria:
        dfw = apply_selection(dfw, precut_criteria)
    dfw = annotate_labels(dfw)

    # 2) Predict both BDTs
    X = dfw[features].values
    dfw[score1_col] = model1.predict_proba(X)[:, 1]
    dfw[score2_col] = model2.predict_proba(X)[:, 1]

    # define flavor masks
    masks_pre = {
        'nue':   dfw['is_nue'],
        'numu':  dfw['is_numuCC'] | dfw['is_numuNC'],
        'nutau': dfw['is_nutau']
    }

    # 3) Pre-BDT rates
    pre_rates = {}
    print("\n=== Pre-BDT rates ===")
    for flavor, mask in masks_pre.items():
        print(f"Calculating rates for flavor: {flavor}")
        raw, rate, unc = compute_rate(dfw[mask], weight_column=weight_col, print_raw=True)
        pre_rates[flavor] = (raw, rate, unc)
    pd.DataFrame.from_dict(pre_rates, orient='index',
                           columns=['raw', 'rate', 'unc']) \
      .to_csv(os.path.join(output_dir, "pre_bdt_rates.csv"))

    # 4) ROC AUC curves
    y_true = dfw['is_nutau'].astype(int).values
    w      = dfw[weight_col].values

    fpr1, tpr1, _ = roc_curve(y_true, dfw[score1_col], sample_weight=w)
    roc_auc1      = auc(fpr1, tpr1)
    fpr2, tpr2, _ = roc_curve(y_true, dfw[score2_col], sample_weight=w)
    roc_auc2      = auc(fpr2, tpr2)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr1, tpr1, label=f"BDT1 (AUC={roc_auc1:.3f})")
    ax_roc.plot(fpr2, tpr2, label=f"BDT2 (AUC={roc_auc2:.3f})")
    ax_roc.plot([0,1], [0,1], 'k--', label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(loc="best")
    fig_roc.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.show()

    # 5) 2D threshold scan & heatmap
    best_t1, best_t2, _ = plot_bdt_threshold_scan(
        dfw, score1_col, score2_col, 'flavor', 'nutau',
        weight_col=weight_col,
        purity_target=purity_target,
        save_path=os.path.join(output_dir, "threshold_scan.png"),
        show=True
    )

    # apply selection
    sel    = (dfw[score1_col] >= best_t1) & (dfw[score2_col] >= best_t2)
    df_sel = dfw[sel]

    # 6) Post-BDT rates
    masks_post = {
        'nue':   df_sel['is_nue'],
        'numu':  df_sel['is_numuCC'] | df_sel['is_numuNC'],
        'nutau': df_sel['is_nutau']
    }
    post_rates = {}
    print("\n=== Post-BDT rates ===")
    for flavor, mask in masks_post.items():
        print(f"Calculating post-BDT rates for flavor: {flavor}")
        raw, rate, unc = compute_rate(df_sel[mask], weight_column=weight_col, print_raw=True)
        post_rates[flavor] = (raw, rate, unc)
    pd.DataFrame.from_dict(post_rates, orient='index',
                           columns=['raw', 'rate', 'unc']) \
      .to_csv(os.path.join(output_dir, "post_bdt_rates.csv"))

    # 7) Score histograms with cut-line legends
    bins = np.linspace(0, 1, 50)
    vline_kwargs = dict(color='k', linestyle='--', linewidth=1)

    # Pre-BDT DataFrames by flavor
    dfs_pre = {fl: dfw[mask] for fl, mask in masks_pre.items()}
    ax1 = plot_histograms(
        data_input=dfs_pre,
        plotvar=score1_col,
        bins=bins,
        weights_map=weight_col,
        xlabel=score1_col,
        ylabel="Weighted counts",
        title=f"{score1_col} distribution by flavor",
        labels=['nue','numu','nutau'],
        show=False
    )
    ax1.axvline(best_t1, **vline_kwargs, label=f"BDT1 cut @ {best_t1:.2f}")
    ax1.legend(loc='best')
    plt.show()
    ax1.figure.savefig(os.path.join(output_dir, f"{score1_col}_hist.png"))

    ax2 = plot_histograms(
        data_input=dfs_pre,
        plotvar=score2_col,
        bins=bins,
        weights_map=weight_col,
        xlabel=score2_col,
        ylabel="Weighted counts",
        title=f"{score2_col} distribution by flavor",
        labels=['nue','numu','nutau'],
        show=False
    )
    ax2.axvline(best_t2, **vline_kwargs, label=f"BDT2 cut @ {best_t2:.2f}")
    ax2.legend(loc='best')
    plt.show()
    ax2.figure.savefig(os.path.join(output_dir, f"{score2_col}_hist.png"))

    # 8) Energy histograms after BDT cut
    energy_var  = 'cscdSBU_MonopodFit4_noDC_energy'
    energy_bins = np.logspace(4.5, 8, 30)
    dfs_post = {fl: df_sel[mask] for fl, mask in masks_post.items()}
    ax3 = plot_histograms(
        data_input=dfs_post,
        plotvar=energy_var,
        bins=energy_bins,
        weights_map=weight_col,
        xscale='log',
        xlabel=energy_var,
        ylabel="Weighted counts",
        title="Energy after BDT cuts by flavor",
        labels=['nue','numu','nutau'],
        show=False
    )
    plt.show()
    ax3.figure.savefig(os.path.join(output_dir, "energy_after_bdt.png"))

    # 9) Optional diagnostics
    if run_diagnostics:
        # 9.1 Score vs Reco Distance (log 2D hist)
        df_nutau = dfw[dfw['is_nutau']]
        x = df_nutau['Taupede_ftp_Distance_value']
        y = df_nutau[score1_col]
        fig, ax = plt.subplots()
        h = ax.hist2d(x, y, bins=50, norm=LogNorm(), cmap='viridis',
                      weights=df_nutau[weight_col])
        ax.set_xlabel("Taupede_ftp_Distance_value")
        ax.set_ylabel(score1_col)
        ax.set_title("Nutau BDT1 Score vs Reco Distance")
        fig.colorbar(h[3], ax=ax, label='Weighted Counts')
        fig.savefig(os.path.join(output_dir, "nutau_score_vs_distance.png"))
        plt.show()

        # 9.2 Purity vs BDT1 cut
        cuts = np.linspace(0,1,101)
        purity_vs_cut = []
        for t in cuts:
            sel_t = dfw[score1_col] >= t
            wsel = dfw.loc[sel_t, weight_col]
            if wsel.sum() == 0:
                purity_vs_cut.append(0.0)
            else:
                wnutau = dfw.loc[sel_t & dfw['is_nutau'], weight_col].sum()
                purity_vs_cut.append(wnutau / wsel.sum())
        fig, ax = plt.subplots()
        ax.plot(cuts, purity_vs_cut, label='Purity vs cut')
        ax.set_xlabel(f"{score1_col} threshold")
        ax.set_ylabel("Nutau purity")
        ax.set_title("Nutau Purity vs BDT1 Score Cut")
        ax.legend(loc='best')
        fig.savefig(os.path.join(output_dir, "purity_vs_cut.png"))
        plt.show()

        # 9.3 Score‐weighted Distance vs Energy (if available)
        energy2 = 'Taupede_ftp_1Particles_energy'
        if energy2 in dfw.columns:
            x = dfw['Taupede_ftp_Distance_value']
            y = dfw[energy2]
            w = dfw[score1_col]
            fig, ax = plt.subplots()
            h = ax.hist2d(x, y, bins=50, norm=LogNorm(), cmap='plasma', weights=w)
            ax.set_xlabel("Taupede_ftp_Distance_value")
            ax.set_ylabel(energy2)
            ax.set_title("Score‐weighted Distance vs Energy")
            fig.colorbar(h[3], ax=ax, label=f"{score1_col} weight")
            fig.savefig(os.path.join(output_dir, "score_vs_distance_vs_energy.png"))
            plt.show()

    # 10) Purity & efficiency
    sig_pre   = pre_rates['nutau'][1]
    sig_post  = post_rates['nutau'][1]
    total_post = sum(r[1] for r in post_rates.values())
    purity     = sig_post / total_post if total_post > 0 else 0.0
    efficiency = sig_post / sig_pre   if sig_pre > 0    else 0.0

    # save summary
    summary = {
        'pre_rates':  pre_rates,
        'best_t1':    best_t1,
        'best_t2':    best_t2,
        'post_rates': post_rates,
        'purity':     purity,
        'efficiency': efficiency
    }
    pd.Series({
        'best_t1': best_t1,
        'best_t2': best_t2,
        'purity': purity,
        'efficiency': efficiency
    }).to_csv(os.path.join(output_dir, "final_summary.csv"))

    print(f"\nFinal purity: {purity:.3f}, efficiency: {efficiency:.3f}")
    return summary
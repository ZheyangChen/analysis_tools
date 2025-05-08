# file: analysis_tools/workflows/BDT_pipeline.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable, Optional, List, Dict

from analysis_tools.my_selectors.apply_selection     import apply_selection
from analysis_tools.BDT_tools.Testset_preparation   import annotate_labels
from analysis_tools.calculators.event_rates          import compute_rate
from analysis_tools.plotters.histogram_plot          import plot_histograms
from analysis_tools.BDT_tools.BDT_evaluation         import plot_bdt_threshold_scan
from analysis_tools.BDT_tools.BDT_training           import model_training,plot_feature_importances
from analysis_tools.workflows.evaluation_flow        import evaluation_flow


def _mask_from_spec(df: pd.DataFrame, spec) -> pd.Series:
    """
    Turn a mask spec into a boolean Series.
      - If spec is a column name (str), return df[spec].
      - If spec is a callable, return spec(df).
    """
    if isinstance(spec, str):
        return df[spec]
    elif callable(spec):
        return spec(df)
    else:
        raise ValueError("mask spec must be str or callable")


def run_bdt_pipeline(
    train_df: pd.DataFrame,
    eval_df:  pd.DataFrame,
    global_precuts: Optional[dict],
    model_configs: List[Dict],
    features:   List[str],
    weight_col: str = "weight",
    purity_target: float = 0.90,
    score_cols:       List[str]  = ["bdt1_score","bdt2_score"],
    train_outdir:     str        = "pipeline_train",
    eval_outdir:      str        = "pipeline_eval"
):
    """
    A general pipeline for training N BDTs and evaluating them.

    Parameters
    ----------
    train_df, eval_df : DataFrames
       Independent train & eval samples.
    global_precuts : dict or None
       Passed to apply_selection() on both sets first.
    model_configs : list of dicts, each with keys:
       - name       : unique string, e.g. "bdt1"
       - sig_mask   : str or callable(df)->bool mask
       - bg_mask    : str or callable
       - model_name : filename prefix for saving
       - params     : dict of XGBoost GridSearchCV params (overrides defaults)
    features : list of column names to use
    weight_col : name of the weight column
    purity_target : float
    score_cols : list of length N, the score‐column names in order
    train_outdir, eval_outdir : output directories
    """
    os.makedirs(train_outdir, exist_ok=True)
    os.makedirs(eval_outdir,  exist_ok=True)

    # 1) Global precuts + labeling on both sets
    if global_precuts:
        train_df = apply_selection(train_df, global_precuts)
        eval_df  = apply_selection(eval_df,  global_precuts)
    train_tag = annotate_labels(train_df)
    eval_tag  = annotate_labels(eval_df)

    trained_models = {}

    # 2) Loop over each model config
    
    for cfg, score_col in zip(model_configs, score_cols):
        name     = cfg['name']
        sig_spec = cfg['sig_mask']
        bg_spec  = cfg['bg_mask']
        params   = cfg.get('params', None)

        # build train subset
        sig_mask = _mask_from_spec(train_tag, sig_spec)
        bg_mask  = _mask_from_spec(train_tag, bg_spec)
        df2 = train_tag[sig_mask | bg_mask].copy()

        # create 'label' column: 1 for sig, 0 for bg
        df2['label'] = sig_mask[df2.index].astype(int)

        # 3) Train & save model in one shot
        save_path = os.path.join(train_outdir, f"{name}_model.pkl")
        model     = model_training(
            df2[features + ['label']],
            save_path=save_path
        )
        trained_models[name] = model
        print(f"Trained {name}, saved to {save_path}")

        # — NEW: plot & save feature importances —
        fi_path = os.path.join(train_outdir, f"{name}_importances.png")
        plot_feature_importances(
            model,
            feature_names=features,
            top_n=len(features),
            title=f"{name} Feature Importances",
            save_path=fi_path
        )
        print(f"Saved feature importances to {fi_path}")


    # 4) Evaluate all trained models together
    # Prepare score labels in eval_tag
    for cfg, score_col in zip(model_configs, score_cols):
        name     = cfg['name']
        model    = trained_models[name]
        eval_tag[score_col] = model.predict_proba(eval_tag[features].values)[:,1]

    summary = evaluation_flow(
        eval_tag,
        *trained_models.values(),      # pass models in same order
        features,
        weight_col=weight_col,
        precut_criteria=None,          # already applied
        purity_target=purity_target,
        score1_col=score_cols[0],
        score2_col=score_cols[1],
        output_dir=eval_outdir
    )

    return trained_models, summary
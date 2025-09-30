# file: analysis_tools/workflows/prediction_flow.py

import pandas as pd
from typing import Dict, List, Optional
import numpy as np

import xgboost as xgb


class BoosterWrapper:
    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict_proba(self, X):
        dmatrix = xgb.DMatrix(X)
        proba = self.booster.predict(dmatrix)
        return np.column_stack((1 - proba, proba))  # shape (n, 2)


def predict_scores(
    df: pd.DataFrame,
    models: Dict[str, object],
    features: List[str],
    score_col_map: Optional[Dict[str,str]] = None
) -> pd.DataFrame:
    """
    Given a DataFrame, a dict of trained classifiers, and the feature list,
    returns a new DataFrame with one new column per model containing the score.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    models : dict
        Mapping from model name (e.g. "bdt1") to a fitted classifier with predict_proba.
    features : list of str
        Columns to feed into each model.
    score_col_map : dict, optional
        If provided, maps model-name -> desired score-column name.  
        Otherwise uses f"{model}_score".

    Returns
    -------
    df_out : pd.DataFrame
        A copy of df with added score columns.
    """
    df_out = df.copy()
    X = df_out[features].values

    score_col_map = score_col_map or {}
    for name, model in models.items():
        col = score_col_map.get(name, f"{name}_score")
        # assume binary classifier, score = P(class=1)
        df_out[col] = model.predict_proba(X)[:,1]

    return df_out


def apply_score_thresholds(
    df: pd.DataFrame,
    thresholds: Dict[str, float]
) -> pd.DataFrame:
    """
    Given a DataFrame with score columns, apply thresholds to produce boolean masks.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns matching the keys of `thresholds`.
    thresholds : dict
        Mapping score-col -> threshold.  
        e.g. {"bdt1_score": 0.5, "bdt2_score": 0.4}

    Returns
    -------
    df_out : pd.DataFrame
        A copy of df with new boolean columns named f"{score_col}_pass"
        containing True where df[score_col] >= threshold.
    """
    df_out = df.copy()
    for score_col, thresh in thresholds.items():
        pass_col = f"{score_col}_pass"
        df_out[pass_col] = df_out[score_col] >= thresh
    return df_out


def prediction_flow(
    df: pd.DataFrame,
    models: Dict[str, object],
    features: List[str],
    thresholds: Optional[Dict[str,float]] = None,
    score_col_map: Optional[Dict[str,str]] = None
) -> pd.DataFrame:
    """
    Full prediction pipeline: compute score columns and (optionally)
apply selection masks.

Parameters
----------
df : pd.DataFrame
models : dict
    e.g. {"bdt1": bdt1_model, "bdt2": bdt2_model}
features : list of str
    Feature column names.
thresholds : dict, optional
    score_col -> threshold (e.g. {"bdt1_score":0.5})
score_col_map : dict, optional
    model_name -> score column name override.

Returns
-------
df_pred : pd.DataFrame
    Original df plus score columns and `<score>_pass` masks.
"""
    # 1) compute scores
    df_pred = predict_scores(df, models, features, score_col_map=score_col_map)

    # 2) apply thresholds if given
    if thresholds:
        df_pred = apply_score_thresholds(df_pred, thresholds)

    return df_pred
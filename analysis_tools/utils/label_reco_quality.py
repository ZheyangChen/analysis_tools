# file: analysis_tools/utils/label_reco_quality.py

import operator
import pandas as pd
from typing import Dict, Tuple, Union

# map string ops to functions
_ops = {
    '>':  operator.gt,
    '<':  operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
}

def label_reco_quality(
    df: pd.DataFrame,
    criteria: Dict[str, Union[float, Tuple[str, float]]],
    new_col: str
) -> pd.Series:
    """
    Build a boolean Series named `new_col` that is True whenever
    all of the given criteria are satisfied.

    Parameters
    ----------
    df : pandas.DataFrame
    criteria : dict
      Keys are column names in `df`.  Values are either
        - a single float (interpreted as >= value), or
        - a tuple (op_str, value), op_str in ['>','<','>=','<=','==','!='].
    new_col : str
      Name to assign to the returned Series.

    Returns
    -------
    mask : pd.Series[bool]
      True if row meets every criterion.
    """
    mask = pd.Series(True, index=df.index)
    for col, cond in criteria.items():
        if isinstance(cond, tuple):
            op_str, val = cond
            func = _ops.get(op_str)
            if func is None:
                raise ValueError(f"Unsupported operator {op_str!r}")
        else:
            # single number → interpret as ">= cond"
            func, val = _ops['>='], cond

        mask &= func(df[col], val)

    return mask.rename(new_col)
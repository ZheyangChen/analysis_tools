# apply_selection.py

"""
apply_selection.py

Provides functions to filter pandas DataFrames (or lists/dicts of DataFrames)
using flexible selection criteria with comparison operators, including AND/OR
combinations on the same column.
"""

import operator
import pandas as pd

def select_data_with_operators(df: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    """
    Filter a DataFrame based on a dictionary of selection criteria with comparison operators.
    
    Supports per-column:
      - AND of tests:    {'x': [('>', -10), ('<', 10)]}
      - OR of tests:     {'y': {'or': [('>', 50), ('<', -50)]}}
      - AND+OR combined:{'x': {'and':[...], 'or':[...]}}

    Parameters
    ----------
    df : pd.DataFrame
    criteria : dict
        column → one of:
          * (op, val) tuple
          * list of (op, val) tuples  → AND
          * dict with keys 'and' and/or 'or', whose values are lists of (op,val)
    """
    # map string→function
    ops = {
        '>':  operator.gt,
        '<':  operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
    }
    
    mask = pd.Series(True, index=df.index)
    
    for col, cond in criteria.items():
        # start with all-True
        col_mask = pd.Series(True, index=df.index)

        # If cond is a dict, it may have 'and' and/or 'or'
        if isinstance(cond, dict):
            # handle AND-list
            if 'and' in cond:
                for op_str, val in cond['and']:
                    if op_str not in ops:
                        raise ValueError(f"Unsupported op {op_str}")
                    col_mask &= ops[op_str](df[col], val)
            # handle OR-list
            if 'or' in cond:
                or_mask = pd.Series(False, index=df.index)
                for op_str, val in cond['or']:
                    if op_str not in ops:
                        raise ValueError(f"Unsupported op {op_str}")
                    or_mask |= ops[op_str](df[col], val)
                col_mask &= or_mask

        else:
            # not a dict: treat as AND of one or more tests
            tests = cond
            # single tuple → wrap in list
            if isinstance(cond, tuple) and len(cond)==2:
                tests = [cond]
            if not isinstance(tests, list):
                raise ValueError(f"Invalid condition for {col}: {cond!r}")
            
            for op_str, val in tests:
                if op_str not in ops:
                    raise ValueError(f"Unsupported op {op_str}")
                col_mask &= ops[op_str](df[col], val)

        mask &= col_mask

    return df[mask]


def apply_selection(data_input, criteria):
    """
    Apply select_data_with_operators to a DataFrame, or to each DataFrame in a list or dict.

    Parameters
    ----------
    data_input : pandas.DataFrame or list of DataFrames or dict of DataFrames
        The input data to filter.
    criteria : dict
        The same criteria dict passed to select_data_with_operators.

    Returns
    -------
    Same type as data_input:
      - DataFrame -> filtered DataFrame
      - list -> list of filtered DataFrames
      - dict -> dict mapping same keys to filtered DataFrames
    """
    if isinstance(data_input, pd.DataFrame):
        return select_data_with_operators(data_input, criteria)

    elif isinstance(data_input, list):
        return [select_data_with_operators(df, criteria) for df in data_input]

    elif isinstance(data_input, dict):
        return {key: select_data_with_operators(df, criteria)
                for key, df in data_input.items()}

    else:
        raise ValueError(
            "data_input must be a pandas DataFrame, "
            "a list of DataFrames, or a dict of DataFrames."
        )

'''
# Example usage:
from apply_selection import apply_selection

# Example 1: single DataFrame, –10 < x < 10 AND (y > 50 OR y < –50)
crit = {
    'x': [('>', -10), ('<', 10)],
    'y': {'or': [('>', 50), ('<', -50)]}
}
filtered_df = apply_selection(df, crit)

# Example 2: list of DataFrames
dfs = [df1, df2]
filtered_list = apply_selection(dfs, {'z': ('>=', 0)})

# Example 3: dict of DataFrames
dct = {'runA': dfA, 'runB': dfB}
filtered_dict = apply_selection(dct, {'energy': [('>', 0), ('<', 100)]})
'''
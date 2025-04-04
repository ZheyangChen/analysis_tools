import operator
import pandas as pd

def select_data_with_operators(df, criteria):
    """
    Filter a DataFrame based on a dictionary of selection criteria with comparison operators.
    
    Each key in `criteria` should be a column name. The corresponding value can be:
      - A single value, in which case an equality check is performed (df[col] == value).
      - A tuple (op, value), where op is a string ('>', '<', '>=', '<=', '==', '!=')
        and value is the threshold. The appropriate comparison is then applied.
      - A list of such tuples or values. In that case, conditions are OR-combined for that column.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to filter.
    criteria : dict
        Dictionary with keys as column names and values as either a constant, a tuple, or
        a list of constants/tuples for filtering.
        
    Returns
    -------
    filtered_df : pandas.DataFrame
        A DataFrame containing only the rows that meet all the selection criteria.
        
    Raises
    ------
    ValueError
        If an unsupported operator is provided.
    """
    # Mapping of operator strings to functions.
    ops = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne
    }
    
    # Start with a mask that selects all rows.
    mask = pd.Series(True, index=df.index)
    
    for col, cond in criteria.items():
        # If cond is a list, OR all sub-conditions together.
        if isinstance(cond, list):
            col_mask = pd.Series(False, index=df.index)
            for sub_cond in cond:
                if isinstance(sub_cond, tuple) and len(sub_cond) == 2:
                    op_str, value = sub_cond
                    if op_str not in ops:
                        raise ValueError(f"Operator '{op_str}' is not supported. Choose from {list(ops.keys())}.")
                    col_mask |= ops[op_str](df[col], value)
                else:
                    col_mask |= (df[col] == sub_cond)
            mask &= col_mask
        # If cond is a tuple, apply the corresponding operator.
        elif isinstance(cond, tuple) and len(cond) == 2:
            op_str, value = cond
            if op_str not in ops:
                raise ValueError(f"Operator '{op_str}' is not supported. Choose from {list(ops.keys())}.")
            mask &= ops[op_str](df[col], value)
        # Otherwise, default to an equality check.
        else:
            mask &= (df[col] == cond)
    
    return df[mask]

'''
# Example usage:
if __name__ == '__main__':
    # Create a sample DataFrame.
    df = pd.DataFrame({
        'A': range(10),
        'B': [5, 3, 8, 10, 2, 7, 6, 4, 9, 1],
        'C': [0, 1, 2, 3, 4, 3, 2, 1, 0, 1]
    })
    
    # Define selection criteria:
    # Select rows where A > 4, B <= 7, and C == 3.
    criteria = {
        'A': ('>', 4),
        'B': ('<=', 7),
        'C': 3
    }
    
    filtered_df = select_data_with_operators(df, criteria)
    print(filtered_df)
'''
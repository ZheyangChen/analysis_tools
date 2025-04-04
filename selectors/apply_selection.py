import operator
import pandas as pd

def select_data_with_operators(df, criteria):
    """
    Filter a DataFrame based on a dictionary of selection criteria with comparison operators.
    
    Each key in `criteria` should be a column name. The corresponding value can be:
      - A single value, in which case an equality check is performed (df[col] == value), or
      - A tuple (op, value), where op is a string ('>', '<', '>=', '<=', '==', '!=') and 
        value is the threshold. The appropriate comparison is then applied.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to filter.
    criteria : dict
        Dictionary with keys as column names and values as either a constant or a tuple 
        (operator, value) for filtering.
        
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
        if isinstance(cond, tuple) and len(cond) == 2:
            op_str, value = cond
            if op_str not in ops:
                raise ValueError(f"Operator '{op_str}' is not supported. Choose from {list(ops.keys())}.")
            op_func = ops[op_str]
            mask &= op_func(df[col], value)
        else:
            # Default to equality if no operator is specified.
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
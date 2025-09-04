from typing import List
import pandas as pd

def rename_prefixes(
    df: pd.DataFrame,
    old_prefixes: List[str],
    new_prefixes: List[str],
    inplace: bool = True,
    mode: str = "replace"  # or "add"
) -> pd.DataFrame:
    """
    Rename—or optionally duplicate—columns by prefix.

    Parameters
    ----------
    df : pd.DataFrame
    old_prefixes : list of str
        Prefixes to look for.
    new_prefixes : list of str
        What to replace them with (same length as old_prefixes).
    inplace : bool, default True
        If True, modify `df` in place and return None; else return a new DataFrame.
    mode : {'replace', 'add'}
        - 'replace': rename matching columns in place (classic .rename).  
        - 'add':    keep the originals and also add new columns under the new names.

    Returns
    -------
    pd.DataFrame or None
      If `inplace=True`, returns None. Otherwise returns the modified copy.
    """
    if mode not in ("replace", "add"):
        raise ValueError("mode must be either 'replace' or 'add'")
    if len(old_prefixes) != len(new_prefixes):
        raise ValueError("old_prefixes and new_prefixes must be the same length")

    # work on a copy if not in place
    target = df if inplace else df.copy()

    # Build the mapping for prefix renames + fixed name changes
    mapping = {}
    for old, new in zip(old_prefixes, new_prefixes):
        for col in target.columns:
            if col.startswith(old):
                mapping[col] = new + col[len(old):]

    # two fixed renames (always applied)
    mapping.update({
        "Taupede_Asymmetry_value":      "Taupede_spice3_Asymmetry_value",
        "Taupede_Distance_value":       "Taupede_spice3_Distance_value",
        "TauMonoDiff_rlogl_value":      "Taupede_spice3MonoDiff_rlogl_value",
    })

    if mode == "replace":
        # simply rename columns
        return target.rename(columns=mapping, inplace=inplace)

    # mode == "add": duplicate each matched column under its new name
    for old_name, new_name in mapping.items():
        # avoid overwriting if new_name already exists
        if new_name in target.columns:
            raise KeyError(f"Column {new_name!r} already exists in DataFrame")
        target[new_name] = target[old_name]

    return None if inplace else target
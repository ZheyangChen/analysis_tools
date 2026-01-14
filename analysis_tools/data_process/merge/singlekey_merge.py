import pandas as pd
import numpy as np
from pathlib import Path
import h5py
import tables

def read_hdf5_with_subfields(file_path, key, subfield_map=None, column_list=None):
    """
    Read scalar + compound subfields from an HDF5 group using PyTables.

    Parameters
    ----------
    file_path : str
        Path to the .h5/.hdf file.
    key : str
        Group name to read.
    subfield_map : dict, optional
        { "OfflineFilterMask": { "OfflineCscd_24": ["condition", "prescale"] } }
    column_list : list, optional
        List of scalar + subfield columns to include.

    Returns
    -------
    pd.DataFrame
    """
    import tables
    with tables.open_file(file_path) as h5:
        node = h5.get_node(f"/{key}")
        arr = node.read()

    df = pd.DataFrame()

    # Extract scalar fields
    scalar_fields = [n for n in arr.dtype.names if arr.dtype[n].shape == ()]
    for name in scalar_fields:
        if (not column_list) or (name in column_list):
            df[name] = arr[name]

    # Extract compound subfields
    normalized_key = key.lstrip('/')
    if subfield_map and normalized_key in subfield_map:
        #print('checking for subfiled keys')
        for struct_name, subfields in subfield_map[normalized_key].items():
            if struct_name not in arr.dtype.names:
                print(f"⚠️ Compound field '{struct_name}' not found in key '{normalized_key}'")
                continue

            #print('struct_name: ', struct_name)
            compound_data = arr[struct_name]
            #print(f"✅ Reading compound field: {struct_name}")
            #print(f"    Shape: {compound_data.shape}, dtype: {compound_data.dtype}")

            try:
                for i, subname in enumerate(subfields):
                    col_name = f"{struct_name}_{subname}"
                    if (not column_list) or (col_name in column_list):
                        df[col_name] = compound_data[:, i]
            except Exception as e:
                print(f"❌ Failed to extract subfields from '{struct_name}': {e}")

    return df.add_prefix(key + "_")

def merge(file_path, key_list, key_column_map=None, subfield_map=None):
    """
    Merge selected keys from one HDF5 file into a single DataFrame.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.
    key_list : list
        Keys to read.
    key_column_map : dict, optional
        {key: [columns]} to keep.
    subfield_map : dict, optional
        {key: {compound: [subfields]}} to extract from structured data.

    Returns
    -------
    pd.DataFrame
    """
    merged = None
    for key in key_list:
        try:
            df = read_hdf5_with_subfields(
                file_path,
                key,
                subfield_map=subfield_map,
                column_list=key_column_map.get(key) if key_column_map else None
            )
            merged = df if merged is None else merged.join(df, how="inner")
        except Exception as e:
            print(f"Skipping key '{key}' in {file_path}: {e}")
    return merged

def flatten_hdf5_structured_dataset(file_path, key, subfield_map=None):
    """
    Flatten a structured HDF5 dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file.
    key : str
        HDF5 dataset key.
    subfield_map : dict, optional
        If provided, only extract listed subfields from compound columns.
        Format: {"compound_column": ["subfield1", "subfield2", ...]}

    Returns
    -------
    pd.DataFrame
    """
    with h5py.File(file_path, 'r') as f:
        data = f[key][:]

    flattened = {}
    subfield_spec = subfield_map.get(key, {}) if subfield_map else {}

    for name in data.dtype.names:
        field = data[name]
        if field.dtype.fields is None:
            flattened[name] = field
        else:
            selected_subfields = subfield_spec.get(name, field.dtype.names)
            for subname in selected_subfields:
                if subname in field.dtype.names:
                    flattened[f"{name}_{subname}"] = field[subname]
                else:
                    print(f"Warning: subfield '{subname}' not found in '{name}' under '{key}'")

    return pd.DataFrame(flattened)


def split_hdf_key(
    hdf_path: Path,
    source_key: str,
    replace_name: str,
    new_key_suffix: tuple = ("_1", "_2"),
    mode: str = "a",
    chunk_size: int = 50000
) -> None:
    
    new_key1 = source_key.replace(replace_name, new_key_suffix[0])
    new_key2 = source_key.replace(replace_name, new_key_suffix[1])
    # Define the key for distances:
    distance_key = f'{replace_name}_Distance_value'
    asymmetry_key = f'{replace_name}_Asymmetry_value'

    # Check that target keys don't already exist.
    with pd.HDFStore(hdf_path, mode='r') as store:
        existing_keys = store.keys()
        if new_key1 in existing_keys or new_key2 in existing_keys or distance_key in existing_keys:
            raise ValueError("One or more target keys already exist")

    row_counter = 0
    
    with pd.HDFStore(hdf_path, mode=mode) as store:
        nrows = store.get_storer(source_key).nrows
        
        for chunk_start in range(0, nrows, chunk_size):
            chunk = store.select(
                key=source_key,
                start=chunk_start,
                stop=chunk_start + chunk_size
            )
            
            # Split rows into two chunks based on position in the chunk.
            mask = (np.arange(len(chunk)) % 2 == 0)
            
            odd_chunk = chunk[mask].reset_index(drop=True)
            even_chunk = chunk[~mask].reset_index(drop=True)
            
            if not odd_chunk.empty:
                odd_chunk.to_hdf(
                    store,
                    key=new_key1,
                    format='table',
                    append=(row_counter > 0),
                    data_columns=True,
                    index=False
                )
                
            if not even_chunk.empty:
                even_chunk.to_hdf(
                    store,
                    key=new_key2,
                    format='table',
                    append=(row_counter > 0),
                    data_columns=True,
                    index=False
                )
            
            # Compute the distance if both chunks have data.
            # (Assumes that odd_chunk and even_chunk have matching rows.)
            if not odd_chunk.empty and not even_chunk.empty:
                # Make sure that the columns are named as follows:
                # For odd_chunk: new_key1 + '_x', new_key1 + '_y', new_key1 + '_z'
                # For even_chunk: new_key2 + '_x', new_key2 + '_y', new_key2 + '_z'
                dist = np.sqrt(
                    (odd_chunk[f"{new_key1}_x"] - even_chunk[f"{new_key2}_x"])**2 +
                    (odd_chunk[f"{new_key1}_y"] - even_chunk[f"{new_key2}_y"])**2 +
                    (odd_chunk[f"{new_key1}_z"] - even_chunk[f"{new_key2}_z"])**2
                )
                # Create a DataFrame with the distance results.
                distance_df = pd.DataFrame({"distance": dist})
                
                distance_df.to_hdf(
                    store,
                    key=distance_key,
                    format='table',
                    append=(row_counter > 0),
                    data_columns=True,
                    index=False
                )

                E_asymmetry = (
                    (odd_chunk[f"{new_key1}_energy"] - even_chunk[f"{new_key2}_energy"]) 
                    /(odd_chunk[f"{new_key1}_energy"] + even_chunk[f"{new_key2}_energy"])
                )
    
                asymmetry_df = pd.DataFrame({"asymmetry": E_asymmetry})
                asymmetry_df.to_hdf(
                    store,
                    key=asymmetry_key,
                    format='table',
                    append=(row_counter > 0),
                    data_columns=True,
                    index=False
                )
            
            row_counter += len(chunk)
        
        store.flush()
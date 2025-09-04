import pandas as pd
import numpy as np
from pathlib import Path
import h5py


def try_read_hdf_flat_or_structured(name, key):
    """
    Attempt to read key from HDF5 file. If structured (fails with pandas), flatten it.
    Returns: DataFrame with column names prefixed by the key.
    """
    try:
        # Try regular read
        df = pd.read_hdf(name, key=key)
    except (ValueError, TypeError):  # Structured compound type
        df = flatten_hdf5_structured_dataset(name, key)
    return df.add_prefix(key + '_')


def merge(name, key_list):
    """
    Merge multiple keys from an HDF5 file into one DataFrame,
    automatically flattening structured compound keys.
    """
    merged = try_read_hdf_flat_or_structured(name, key_list[0])
    for key in key_list[1:]:
        try:
            df = try_read_hdf_flat_or_structured(name, key)
            merged = merged.join(df, how='inner')
        except Exception as e:
            print(f"Skipping key '{key}' due to error: {e}")
    return merged


def flatten_hdf5_structured_dataset(file_path, key):
    """
    Flatten a structured HDF5 dataset into a pandas DataFrame.
    """
    with h5py.File(file_path, 'r') as f:
        data = f[key][:]

    flattened = {}
    for name in data.dtype.names:
        field = data[name]
        if field.ndim == 1:
            flattened[name] = field
        else:
            for i in range(field.shape[1]):
                flattened[f'{name}_{i}'] = field[:, i]

    df = pd.DataFrame(flattened)
    return df

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
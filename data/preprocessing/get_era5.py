import glob, h5py, warnings
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from configs.config import cfg
warnings.filterwarnings('ignore')

def main(start_date: str, end_date: str, output_name: str) -> None:
    all_file_path = sorted(glob.glob(f"{cfg.ERA5_ROOT_PATH}*.nc"))
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.Timedelta(days=1)
    all_timestamps = (end_dt - start_dt).days * 24
    print(f"Estimated time step: {all_timestamps}")
    timestep, idx = [], 0

    with h5py.File(output_name, "w") as hf:
        # Create Datasets within the Group, Shape: (Time, n_data, 1)
        x_dset = hf.create_dataset('val', shape=(all_timestamps, 7, 40, 20), 
                                   dtype='f4', chunks=(1, 7, 40, 20), compression='lzf', fillvalue=np.nan)
        
        for file in tqdm(all_file_path, desc="Preprocessing ERA5 Data ..."):
            ds_data = xr.open_dataset(file)
            
            # Vectorized time parsin, date filtering and save native hourly timestamps (UTC)
            valid_times = pd.to_datetime(ds_data["valid_time"].values)
            valid_mask = (valid_times >= start_dt) & (valid_times < end_dt)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:    continue
            timestep.extend(valid_times[valid_indices].strftime('%Y%m%d%H%M'))
            
            # Vectorized feature loading (Batch Process), shape: (T_valid, 7, 40, 20)
            batch_data = np.stack([ds_data[fea].values[valid_indices].astype(np.float32) for fea in cfg.ERA5_ABBRE], axis=1)
                
            # Batch write to HDF5
            n_valid = len(valid_indices)
            x_dset[idx : idx + n_valid] = batch_data
            idx += n_valid
            
        hf.create_dataset('timestamps', data=np.array(timestep, dtype='S')) # UT time
        lon = ds_data["longitude"].values.astype(np.float32)  # (20,)
        lat = ds_data["latitude"].values.astype(np.float32)   # (40,)
        lon, lat = np.meshgrid(lon, lat)
        hf.create_dataset('geo', data=(lon, lat), dtype='f4') # shape = (2, 40, 20)
        x_dset.resize((idx, 7, 40, 20))
        print(f"Total time steps: {idx}")

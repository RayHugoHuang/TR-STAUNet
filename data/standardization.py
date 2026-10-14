import h5py, os
import numpy as np
import pandas as pd

def compute_and_save_stats(gems_path, era5_path, output_path, gt_path=None, start_date=None, end_date=None):
    """ Computes mean/std/max/min for preprocessed data. Saves as a structured H5 file for direct key-lookup. """
    if os.path.exists(output_path):
        print(f"'{os.path.basename(output_path)}' already exists.")
        return
    print(f"Processing stats from:\n GEMS: {gems_path}\n ERA5: {era5_path}\n GT: {gt_path}")
    # Helper to get valid time indices
    def get_valid_idx(src):
        if start_date and end_date and 'timestamps' in src:
            ts = pd.to_datetime(src['timestamps'][:].astype('S').astype('U'), format='%Y%m%d%H%M')
            mask = (ts >= pd.to_datetime(start_date)) & (ts <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
            return np.where(mask)[0]
        return slice(None)
    
    with h5py.File(output_path, "w") as dst:
        # --- Part 1: GEMS ---
        gems_grp = dst.create_group("GEMS")
        with h5py.File(gems_path, "r") as src:
            idx = get_valid_idx(src) # filter the period
            lens = src['len'][:]
            if isinstance(idx, np.ndarray):
                valid_ts_indices = [t for t in idx if lens[t] > 0]
            else:
                valid_ts_indices = np.where(lens > 0)[0]

            print(f"Computing GEMS Stats across {len(valid_ts_indices)} valid timestamps...")
            C = len(src.attrs['channels'])
            sums, sum_sqs = np.zeros(C, dtype=np.float64), np.zeros(C, dtype=np.float64)
            counts = np.zeros(C, dtype=np.int64)
            mins, maxs = np.full(C, np.inf, dtype=np.float32), np.full(C, -np.inf, dtype=np.float32)

            for t in valid_ts_indices:
                n = int(lens[t])
                if n == 0: continue
                v = src['val'][t, :n, :] # (n, C)
                mask = ~np.isnan(v)
                v_clean = np.where(mask, v, 0.0)

                sums += v_clean.sum(axis=0)
                sum_sqs += (v_clean ** 2).sum(axis=0)
                counts += mask.sum(axis=0)
                mins = np.minimum(mins, np.where(mask, v, np.inf).min(axis=0))
                maxs = np.maximum(maxs, np.where(mask, v, -np.inf).max(axis=0))

            means = (sums / np.maximum(counts, 1)).astype(np.float32)
            stds = np.sqrt(np.maximum((sum_sqs / np.maximum(counts, 1)) - (means.astype(np.float64) ** 2), 0.0)).astype(np.float32)
            stds[stds == 0] = 1.0
            
            gems_grp.create_dataset("mean", data=means)
            gems_grp.create_dataset("std", data=stds)
            gems_grp.create_dataset("max", data=maxs)
            gems_grp.create_dataset("min", data=mins)
            gems_grp.attrs['channels'] = src.attrs['channels']

        # --- Part 2: ERA5 ---
        era5_grp = dst.create_group("ERA5")
        with h5py.File(era5_path, "r") as src:
            idx = get_valid_idx(src) # filter the period
            # ERA5 shape is typically (Time, Channels=7, H=40, W=20)
            data = src['val'][:][idx]
            
            # Calculate stats per channel (Axis 1); Time(0), H(2), W(3) -> Result shape (7,)
            mean = np.nanmean(data, axis=(0, 2, 3)).astype(np.float32)
            std = np.nanstd(data, axis=(0, 2, 3)).astype(np.float32)
            std[std == 0] = 1.0
            
            era5_grp.create_dataset("mean", data=mean)
            era5_grp.create_dataset("std", data=std)
            era5_grp.create_dataset("max", data=np.nanmax(data, axis=(0, 2, 3)).astype(np.float32))
            era5_grp.create_dataset("min", data=np.nanmin(data, axis=(0, 2, 3)).astype(np.float32))

        # --- Part 3: GT ---
        if gt_path is None: return
        gt_grp = dst.create_group("GT")
        with h5py.File(gt_path, "r") as src:
            idx = get_valid_idx(src) # filter the period
            # (Time, Station, Feature=6)
            gt = src["data"][idx]
            mean = np.nanmean(gt, axis=(0, 1)).astype(np.float32)
            std = np.nanstd(gt, axis=(0, 1)).astype(np.float32)
            std[std == 0] = 1.0
            gt_grp.create_dataset("mean", data=mean)
            gt_grp.create_dataset("std", data=std)
            gt_grp.create_dataset("max", data=np.nanmax(gt, axis=(0, 1)).astype(np.float32))
            gt_grp.create_dataset("min", data=np.nanmin(gt, axis=(0, 1)).astype(np.float32))

    print(f"Statistics saved to {output_path}")

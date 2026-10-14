import os, re, glob, warnings, h5py
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional
from configs.config import cfg
warnings.filterwarnings('ignore')
GEMS_DATA, GEMS_DATA_2025 = cfg.GEMS_DATA, cfg.GEMS_DATA_2025
GEMS_FEATURE, GEMS_QUALITY_FLAGS = cfg.GEMS_FEATURE, cfg.GEMS_QUALITY_FLAGS

class GEMSProcessor:
    def __init__(self, lat_range: Tuple[float, float] = cfg.GEO_AREA[1], lon_range: Tuple[float, float] = cfg.GEO_AREA[0]):
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range

    def get_base_grid(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """ Extract base grid and Area of Interest (AOI) mask from the specified NetCDF file. """
        try:
            with xr.open_dataset(file_path, group='/Geolocation Fields') as ds_geo:
                lons = ds_geo["Longitude"].values # (M, N)
                lats = ds_geo["Latitude"].values  # (M, N)

            valid_coords = (~np.isnan(lons)) & (~np.isnan(lats))
            inside_aoi = (lons >= self.lon_min) & (lons < self.lon_max) & \
                         (lats >= self.lat_min) & (lats < self.lat_max)
            
            return lons, lats, inside_aoi & valid_coords
        except (OSError, KeyError, ValueError) as e:
            print(f"Error reading base grid from {file_path}: {e}")
            return None, None, None

    def get_features_from_file(self, file_path: str, f_key: str, var_names: List[str], 
                                   aoi_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """ Extracts multiple features from a single file to minimize I/O overhead. """
        results, flags_mask = {}, None
        # Static mapping for optimal I/O routing
        geo_specific_vars = {"SolarZenithAngle", "ViewingZenithAngle"}
        data_vars = [v for v in var_names if v not in geo_specific_vars]
        geo_vars = [v for v in var_names if v in geo_specific_vars]
        
        if data_vars:
            try:
                with xr.open_dataset(file_path, group='/Data Fields') as ds_data:
                    # Process quality flags once per file
                    if "FinalAlgorithmFlags" in ds_data.variables:
                        flags = ds_data["FinalAlgorithmFlags"].values
                        flags_int = np.nan_to_num(flags, nan=0).astype(np.int32)
                        valid_flags = GEMS_QUALITY_FLAGS.get(f_key, 0)
                        flags_mask = (flags_int & ~valid_flags) == 0
                    
                    for var in data_vars:
                        if var in ds_data.variables:
                            raw_vals = ds_data[var].values.astype(np.float32)
                            fill_val = ds_data[var].attrs.get('_FillValue', -999)
                            valid_min = ds_data[var].attrs.get('valid_min')
                            valid_max = ds_data[var].attrs.get('valid_max')
                            results[var] = self._process_raw_vals(raw_vals, fill_val, flags_mask, aoi_mask, var, valid_min, valid_max)
            except (OSError, KeyError) as e:
                print(f"Error reading Data Fields in {file_path}: {e}")

        if geo_vars:
            try:
                with xr.open_dataset(file_path, group='/Geolocation Fields') as ds_geo:
                    for var in geo_vars:
                        if var in ds_geo.variables:
                            raw_vals = ds_geo[var].values.astype(np.float32)
                            fill_val = ds_geo[var].attrs.get('_FillValue', -999)
                            valid_min = ds_geo[var].attrs.get('valid_min')
                            valid_max = ds_geo[var].attrs.get('valid_max')
                            results[var] = self._process_raw_vals(raw_vals, fill_val, None, aoi_mask, var, valid_min, valid_max)
            except (OSError, KeyError) as e:
                print(f"Error reading geolocation fields in {file_path}: {e}")
        return results

    def _process_raw_vals(self, raw_vals: np.ndarray, fill_val: float, flags_mask: Optional[np.ndarray], aoi_mask: np.ndarray, 
                          feature_name: str, valid_min: Optional[float] = None, valid_max: Optional[float] = None) -> np.ndarray:
        """
        Processes raw values: applies fill values, validity masks, and executes feature-specific extraction.
        """
        if fill_val is not None:    raw_vals[raw_vals == fill_val] = np.nan
        if valid_min is not None:   raw_vals[raw_vals < float(valid_min)] = np.nan
        if valid_max is not None:   raw_vals[raw_vals > float(valid_max)] = np.nan
            
        # Determine valid elements across channels if necessary
        if raw_vals.ndim == 3:
            valid_val = ~np.isnan(raw_vals).all(axis=0)
        else:
            valid_val = ~np.isnan(raw_vals)
            
        # Combine spatial, physical, and quality masks
        valid_mask = aoi_mask & valid_val
        if flags_mask is not None:
            valid_mask &= flags_mask
            
        n_aoi = np.sum(aoi_mask)
        valid_in_aoi = valid_mask[aoi_mask]
        
        # Feature-specific extraction logic (Extensible pattern)
        is_radiance = feature_name == "NormalizedRadiance" and raw_vals.ndim == 3 and raw_vals.shape[0] == 6
        out_vals = np.full((n_aoi, 6 if is_radiance else 1), np.nan, dtype=np.float32)

        if is_radiance:
            out_vals[valid_in_aoi] = raw_vals[:, valid_mask].T
        elif raw_vals.ndim == 3:
            # Determine target channel index: 443nm(3) for reflectance, 443nm(1) for AOD
            ch_idx = 3 if (feature_name == "SurfaceReflectance" and raw_vals.shape[0] >= 4) else 1
            out_vals[valid_in_aoi, 0] = raw_vals[ch_idx, valid_mask]
        else:
            out_vals[valid_in_aoi, 0] = raw_vals[valid_mask]

        return out_vals

def get_files_map(feature_dict: Dict[str, List[str]], start_date: str, end_date: str) -> Dict[str, List[str]]:
    """ Scans the file system to build a mapping of unified timestamps to available file paths. """
    files_map = {}
    dates = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y%m%d').tolist()
        
    print("Scanning file system for GEMS data...")
    for date_str in dates:
        year_month, day = date_str[:6], date_str[6:8]
        root_path = GEMS_DATA if int(year_month[:4]) < 2025 else GEMS_DATA_2025
        
        for feature in feature_dict.keys():
            pattern = os.path.join(root_path, f'L2_{feature}', year_month, day, '*.nc')
            files = glob.glob(pattern)
            ori_files = [f for f in files if 'ORI' in f]
            target_files = ori_files if (ori_files and feature != "SO2") else files
            
            for fpath in target_files:
                # Robustly extract YYYYMMDD_HHMM from standard GEMS filename convention
                match = re.search(r'_(\d{8})_(\d{4})', os.path.basename(fpath))
                if match:
                    ts = f"{match.group(1)}{match.group(2)}"  # YYYYMMDDHHMM
                    files_map.setdefault(ts, []).append(fpath)
                    
    return files_map # ts: filepath

def main(start_date: str, end_date: str, output_h5: str) -> None:
    processor = GEMSProcessor()
    files_map = get_files_map(GEMS_FEATURE, start_date, end_date)
    sorted_ts = sorted(files_map.keys())
    if not sorted_ts:
        print("No valid GEMS files found for the given date range.")
        return

    # 1. Define unified channel layout and allocation map
    ch_names = []
    var_to_col_idx = {}
    curr_idx = 0
    for f_key, var_names in GEMS_FEATURE.items():
        for var in var_names:
            width = 6 if var == "NormalizedRadiance" else 1
            var_to_col_idx[var] = (curr_idx, width)
            curr_idx += width
            
            if var == "NormalizedRadiance":
                ch_names.extend([f"{var}_{i+1}" for i in range(6)])
            else:
                ch_names.append(var)
    total_channels = len(ch_names)

    # 2. Determine maximum possible AOI points to prevent HDF5 bounds truncation
    max_n = 0
    exclude_ts = set()
    for i, ts in enumerate(tqdm(sorted_ts, desc="AOI Base Grid Scanning")):
        for fpath in files_map[ts]:
            lons, lats, aoi_mask = processor.get_base_grid(fpath)
            if aoi_mask is not None:
                pts = int(np.sum(aoi_mask))
                if pts > 5000:
                    print(f"\n[High-resolution mesh] {os.path.basename(fpath)} | Matrix size: {lons.shape} | AOI points: {pts}")
                    exclude_ts.add(ts)
                    if i > 0:
                        exclude_ts.add(sorted_ts[i-1])
                    break
                max_n = max(max_n, pts)
                break  # Single valid grid per timestamp is sufficient
            
    sorted_ts = [ts for ts in sorted_ts if ts not in exclude_ts]
    if max_n == 0:
        print("No valid AOI points found in any file.")
        return
    print(f"Time steps: {len(sorted_ts)}, Total Channels: {total_channels}, Max AOI Points: {max_n}")

    # 3. Create and populate HDF5
    with h5py.File(output_h5, 'w') as hf:
        hf.create_dataset("timestamps", data=np.array(sorted_ts, dtype="S"))
        kwargs = {"dtype":"f4", "compression":"lzf", "fillvalue":np.nan}
        hf.create_dataset('val', shape=(len(sorted_ts), max_n, total_channels), chunks=(1, max_n, total_channels), **kwargs)
        hf.create_dataset('geo', shape=(len(sorted_ts), max_n, 2), chunks=(1, max_n, 2), **kwargs)
        hf.create_dataset('len', shape=(len(sorted_ts),), dtype='i4')
        hf.attrs['channels'] = np.array(ch_names, dtype='S')

        for idx, ts in enumerate(tqdm(sorted_ts, desc="Processing Timestamps")):
            files = files_map[ts]
            aoi_mask = None
            for fpath in files:
                lons, lats, aoi_mask = processor.get_base_grid(fpath)
                if aoi_mask is not None:    break

            if aoi_mask is None:
                hf["len"][idx] = 0
                continue
            
            n_aoi = np.sum(aoi_mask)
            limit = min(n_aoi, max_n)
            # Pre-allocate matrix to accept unified features
            val_matrix = np.full((n_aoi, total_channels), np.nan, dtype=np.float32)
            
            # Batch extract features per file to heavily optimize I/O
            for f_key, var_names in GEMS_FEATURE.items():
                file_path = next((p for p in files if f'L2_{f_key}' in p), None)
                if file_path:
                    extracted_features = processor.get_features_from_file(file_path, f_key, var_names, aoi_mask)
                    for var in var_names:
                        vals = extracted_features.get(var)
                        if vals is not None:
                            col_idx, width = var_to_col_idx[var]
                            val_matrix[:, col_idx : col_idx+width] = vals

            # 3. Store to unified HDF5 tensors. geo: (T, max_n, C=(lon,lat)), val: (T, max_n, total_channels)
            hf['geo'][idx, :limit, 0] = lons[aoi_mask][:limit]
            hf['geo'][idx, :limit, 1] = lats[aoi_mask][:limit]
            hf['val'][idx, :limit, :] = val_matrix[:limit, :]
            hf['len'][idx] = limit
        print(f"Successfully processed and saved to {output_h5}.")
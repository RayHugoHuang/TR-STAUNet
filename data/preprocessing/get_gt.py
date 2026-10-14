import os, glob, h5py
import numpy as np
import pandas as pd
from configs.config import cfg
TARGETFEATURES = cfg.TARGETFEATURES

class SparseTargetBuilder:
    def __init__(self, lat_range=cfg.GEO_AREA[1], lon_range=cfg.GEO_AREA[0], res=cfg.RESOLU):
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range
        self.res = res
        self.lat_grid = np.arange(self.lat_min, self.lat_max, self.res)
        self.lon_grid = np.arange(self.lon_min, self.lon_max, self.res)
        self.height, self.width = len(self.lat_grid), len(self.lon_grid)
        
        self.features = TARGETFEATURES
        self.feature_map = {f: i for i, f in enumerate(self.features)}

    def build(self, site_csv, gt_root_dir, output_h5, start_date="2021010100", end_date="2024123123"):
        # --- Step 1: Topology Mapping ---
        print("Mapping station topology...")
        df_site = pd.read_csv(site_csv)
        
        station_map = {} # {name: tensor_index}
        station_coords = [] # Changed from grid_indices to station_coords

        for _, row in df_site.iterrows():
            name = row.iloc[0] # Station Name
            lon, lat = float(row.iloc[6]), float(row.iloc[7])
            
            if (self.lon_min <= lon <= self.lon_max) and (self.lat_min <= lat <= self.lat_max):
                r = int(round((lat - self.lat_min) / self.res))
                c = int(round((lon - self.lon_min) / self.res))
                if 0 <= r < self.height and 0 <= c < self.width:
                    station_map[name] = len(station_coords)
                    station_coords.append([lon, lat]) # (lon, lat)

        station_coords = np.array(station_coords, dtype=np.float32)
        num_stations = len(station_coords)
        print(f"Valid stations: {num_stations}")

        # --- Step 2: Time Indexing ---
        st_date = pd.to_datetime(start_date, format="%Y%m%d%H")
        ed_date = pd.to_datetime(end_date, format="%Y%m%d%H")
        timestamps = pd.date_range(start=st_date, end=ed_date, freq='h')
        total_hours = len(timestamps)
        
        # Create a fast lookup for timestamps (YYYYMMDDHH -> int index)
        time_index_map = {t: i for i, t in enumerate(timestamps.strftime('%Y%m%d%H'))}
        
        # Output Buffer
        sparse_tensor = np.full((total_hours, num_stations, 6), np.nan, dtype=np.float16)

        # --- Step 3: Vectorized Parsing ---
        print("Processing CSVs (Vectorized)...")
        csv_files = glob.glob(os.path.join(gt_root_dir, "**/*.csv"), recursive=True)
        if len(csv_files) == 0:
            raise ValueError("No CSV files found in the specified directory.")
            
        for fpath in csv_files:
            fname = os.path.basename(fpath)
            st_name = fname.split('_')[0] # e.g., "二林" from "二林_2022.csv"
            
            if st_name not in station_map: continue
            s_idx = station_map[st_name]

            try:
                # 1. Read CSV (Optimized)
                # Use C engine, specify dtypes
                df = pd.read_csv(fpath, usecols=lambda x: x in ['日期', '測項'] or x.isdigit(),
                                 na_values=['#', 'NR', '*', 'x', 'A'], engine='c')
                
                # 2. Filter Target Features
                df = df[df['測項'].isin(self.features)]
                if df.empty: continue

                # 3. Melt (Wide to Long) -> [Date, Item, Hour, Value]
                # Columns: 日期, 測項, 00, 01...
                df_melt = df.melt(id_vars=['日期', '測項'], var_name='Hour', value_name='Value')
                
                # 4. Vectorized Time Construction
                # Date format: "2022/01/01 00:00:00" -> Combine Date + Hour -> "2022010100"
                df_melt['TimeStr'] = df_melt['日期'].str.replace('/', '').str.slice(0, 8) + df_melt['Hour']
                
                # 5. Map to Indices
                # Filter valid times (within range)
                df_melt = df_melt[df_melt['TimeStr'].isin(time_index_map)]
                if df_melt.empty: continue

                # Get Time Indices
                t_indices = df_melt['TimeStr'].map(time_index_map).values
                # Get Feature Indices
                f_indices = df_melt['測項'].map(self.feature_map).values
                
                # 6. Assign to Tensor (Batch Assignment)
                vals = pd.to_numeric(df_melt['Value'], errors='coerce').values
                
                # sparse_tensor[t, s, f] = val
                # Mask out NaNs to allow numpy indexing
                valid_mask = ~np.isnan(vals)
                sparse_tensor[t_indices[valid_mask], s_idx, f_indices[valid_mask]] = vals[valid_mask]

            except Exception as e:
                print(f"Error {fname}: {e}")

        # --- Step 4: Save ---
        print(f"Saving {output_h5}...")
        str_timestamps = timestamps.strftime('%Y%m%d%H00').values.astype('S') # yyyymmddHHMM
        
        with h5py.File(output_h5, 'w') as f:
            f.create_dataset("data", data=sparse_tensor, dtype='float32', chunks=(1, num_stations, 6))
            f.create_dataset("geo", data=station_coords, dtype='float32')  # (lon, lat)
            f.create_dataset("timestamps", data=str_timestamps)
            f.create_dataset("columns", data=np.array(self.features, dtype='S'))
        print("Done.")

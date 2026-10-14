import torch, h5py, random, math
import pandas as pd
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from collections import defaultdict
from torch.utils.data import Subset
from configs.config import cfg
GEMS_FEATURE, ERA5_ABBRE, SEQ_LEN = cfg.GEMS_FEATURE, cfg.ERA5_ABBRE, cfg.SEQ_LEN

# --- Configuration & Constants & Features filtering ---
GEMS_exclude = [cfg.ABBRE_TRANS[k] for k in cfg.EXCLUDE_SET if k in cfg.ABBRE_TRANS] # Translate abbreviation to full name
GEMS_FEA = [v for var in GEMS_FEATURE.values() for v in var if v not in GEMS_exclude]
ERA5_FEA = [fea for fea in ERA5_ABBRE if fea not in cfg.EXCLUDE_SET]

class SparseDataGenerator(Dataset):
    def __init__(self, gems_h5, era_h5, gt_h5, start_date, end_date, meanstd_path=None):
        self.gems_h5 = gems_h5
        self.era5_h5 = era_h5
        self.gt_h5 = gt_h5
        self.file_handles = {}  # Cache for file handles (GEMS, ERA5)
        self.zero_padding = True
        self.active_time = [f for f in cfg.TIME_FEATURE if f in cfg.FEATURES]
        self.active_pos = [f for f in cfg.POSITION_FEATURE if f in cfg.FEATURES]

        # 0. Verify feature sequence strictly matches Config
        self.active_targets = [f for f in cfg.TARGETFEATURES if f not in cfg.EXCLUDE_SET]
        self.gt_keep_idx = torch.tensor([cfg.TARGETFEATURES.index(f) for f in self.active_targets], dtype=torch.long)

        generated_features = (
            [f for f in cfg.GEMS_ABBRE if f not in cfg.EXCLUDE_SET] +
            [f for f in cfg.ERA5_ABBRE if f not in cfg.EXCLUDE_SET] +
            self.active_time + self.active_pos + self.active_targets +
            [f for f in cfg.STATIC_ABBRE if f not in cfg.EXCLUDE_SET]
        )
        if generated_features != cfg.FEATURES:
            raise ValueError(f"Feature sequence mismatch!\nDataLoader: {generated_features}\nConfig:   {cfg.FEATURES}")
        
        # 1. Prepare Timestamps (String) & Indices
        # GEMS timestamps (UTC 45m -> UTC+1h) & ERA5 timestamps (native hourly UTC)
        self.timestamps, self.gems_ts_map = self._load_timestamps_and_map(self.gems_h5, shift_minutes=15)
        _, self.era5_ts_map = self._load_timestamps_and_map(self.era5_h5)

        # GT map for target lookup (UTC+8h -> UTC)
        _, self.gt_ts_map = self._load_timestamps_and_map(self.gt_h5, shift_hours=-8)
        
        # 2. Pre-calculate GEMS index filtering logic
        with h5py.File(self.gems_h5, "r") as hf:
            all_channels = [ch.decode("utf-8") for ch in hf.attrs['channels']]
        self.gems_keep_idx = torch.tensor([all_channels.index(f) for f in GEMS_FEA], dtype=torch.long)
        
        # 3. Load Static Geometry & Preload Target & position embedding
        with h5py.File(self.era5_h5, "r") as hf:
            self.era5_geo_flat = torch.from_numpy(hf['geo'][:]).permute(1, 2, 0).reshape(-1, 2).float() # (2, 40, 20) -> (800, 2)
            if self.active_pos:
                pos_idx = [cfg.POSITION_FEATURE.index(f) for f in self.active_pos]
                self.pos_emd = self.era5_geo_flat[:, pos_idx] - torch.tensor([121.0, 23.5])[pos_idx]

        with h5py.File(self.gt_h5, "r") as hf:
            self.y_target = torch.from_numpy(hf["data"][:]).float() # (T_gt, N_stations, 6)
            self.y_geo = torch.from_numpy(hf["geo"][:]).float()     # (N_stations, 2)
        
        # 4. Generate Valid Sample Indices
        self.sample_indices = self._generate_sample_indices(self.timestamps, start_date, end_date)
        
        # 5. ERA5 Feature Selection Indices
        self.era5_keep_idx = torch.tensor([ERA5_ABBRE.index(f) for f in ERA5_FEA], dtype=torch.long)
        self.uv_idx = torch.tensor([ERA5_ABBRE.index("u10"), ERA5_ABBRE.index("v10")], dtype=torch.long)

        # 6. Pre-slice and cache statistics to avoid repetitive operations in __getitem__
        keep_indices = {'GEMS': self.gems_keep_idx, 'ERA5': self.era5_keep_idx}
        self.stats = self._load_statistics(meanstd_path, keep_indices)
        
        # 7. Pre-standardize the entire GT array for Historical injection
        self.y_target_std = (self.y_target - self.stats['GT']['mean'][:]) / self.stats['GT']['std'][:]

        # 8. Pre-allocate static NaN tensors
        self.nan_y = torch.full((self.y_geo.shape[0], 6), float("nan"))
        self.nan_uv = torch.full((800, 2), float("nan"))
        self.nan_era5 = torch.full((800, len(self.era5_keep_idx)), float("nan"))

    def _load_statistics(self, path, keep_indices):
        stats = {}
        with h5py.File(path, "r") as f:
            for key in ['GEMS', 'ERA5', 'GT']:
                idx = keep_indices.get(key, slice(None))
                stats[key] = {k: torch.from_numpy(f[key][k][idx]) for k in ["mean", "std"]}
        return stats

    def _load_timestamps_and_map(self, path, shift_minutes=0, shift_hours=0):
        with h5py.File(path, "r") as hf:
            raw_ts = hf['timestamps'][:].astype('S').astype('U')
            dt = pd.to_datetime(raw_ts, format="%Y%m%d%H%M")
            dt += pd.Timedelta(minutes=shift_minutes, hours=shift_hours)
            ts_list = dt.strftime('%Y%m%d%H%M').tolist()
            return ts_list, {t: i for i, t in enumerate(ts_list)}

    def _generate_sample_indices(self, timestamps, start, end):
        """ Generates indices for valid target timestamps within the date range. """
        # Timestamps are 'YYYYMMDDHHMM', start/end are 'YYYYMMDD'
        valid_pos = [i for i, ts in enumerate(timestamps) if start <= ts[:8] <= end]
        if not valid_pos:
            raise ValueError("No data found in the specified date range.")
        return valid_pos

    def _get_file_handle(self, key, path):
        if key not in self.file_handles:
            self.file_handles[key] = h5py.File(path, 'r', swmr=True)
        return self.file_handles[key]

    def _get_gems_data(self, hf_gems, ts_str):
        if ts_str in self.gems_ts_map:
            t_idx = self.gems_ts_map[ts_str]
            n_pts = int(hf_gems['len'][t_idx])
            if n_pts > 0:
                return (torch.from_numpy(hf_gems['val'][t_idx, :n_pts])[:, self.gems_keep_idx],
                        torch.from_numpy(hf_gems['geo'][t_idx, :n_pts]))
        return torch.empty((0, len(self.gems_keep_idx))), torch.empty((0, 2))
    
    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        hf_gems = self._get_file_handle('gems', self.gems_h5)
        hf_era5 = self._get_file_handle('era5', self.era5_h5)

        # 1. Determine Temporal Sequence
        target_dt = pd.to_datetime(self.timestamps[self.sample_indices[idx]], format="%Y%m%d%H%M")
        # Historical(t-SEQ_LEN+1 to t) & Forecasting(t+1 to t+cfg.T_LEN)
        seq_dts = [target_dt - pd.Timedelta(hours=i) for i in reversed(range(SEQ_LEN))]
        seq_ts_strs = [dt.strftime('%Y%m%d%H%M') for dt in seq_dts]
        future_ts_strs = [(target_dt + pd.Timedelta(hours=t+1)).strftime('%Y%m%d%H%M') for t in range(cfg.T_LEN)]
        
        # 2. Retrieve Target (GT) & UV data for the forecasting timestamp
        y_val = torch.stack([self.y_target[self.gt_ts_map[ts]] if ts in self.gt_ts_map else self.nan_y for ts in future_ts_strs])
        uv_data = torch.stack([
            torch.from_numpy(hf_era5['val'][self.era5_ts_map[ts]][self.uv_idx, ...]).permute(1, 2, 0).reshape(-1, 2) 
            if ts in self.era5_ts_map else self.nan_uv for ts in future_ts_strs
        ])

        # 3. Process Historical Sequence Data (GEMS, ERA5, Time, Pos, GT)
        # Find the first available valid dt in the current sequence
        first_valid_idx = len(seq_ts_strs) - 1 
        for i, ts_str in enumerate(seq_ts_strs):
            if self._get_gems_data(hf_gems, ts_str)[0].numel() > 0:
                first_valid_idx = i
                break
        
        # Find substitute data starting from first_valid_dt - 24h
        sub_v, sub_g = torch.empty((0, len(self.gems_keep_idx))), torch.empty((0, 2))
        if not self.zero_padding:
            base_sub_dt = seq_dts[first_valid_idx] - pd.Timedelta(hours=24)
            for h in range(24): # Search t-24, t-23, t-22...
                sub_v, sub_g = self._get_gems_data(hf_gems, (base_sub_dt + pd.Timedelta(hours=h)).strftime('%Y%m%d%H%M'))
                if sub_v.numel() > 0:
                    break
        
        frames_val, frames_geo, era5_frames_val, hgt_frames_val = [], [], [], []
        for i, ts_str in enumerate(seq_ts_strs):
            # --------------- GEMS ---------------
            v, g = self._get_gems_data(hf_gems, ts_str)
            if v.numel() == 0 and not self.zero_padding:
                v, g = sub_v, sub_g
            v = (v - self.stats['GEMS']['mean']) / self.stats['GEMS']['std']
            frames_val.append(v)
            frames_geo.append(g)

            # --------------- ERA5 & Time & Pos & Historical GT ---------------
            val_flat = self.nan_era5
            if ts_str in self.era5_ts_map:
                val_grid = torch.from_numpy(hf_era5['val'][self.era5_ts_map[ts_str]])
                val_flat = val_grid[self.era5_keep_idx].permute(1, 2, 0).reshape(-1, len(self.era5_keep_idx))
                val_flat = (val_flat - self.stats['ERA5']['mean']) / self.stats['ERA5']['std']
            
            if self.active_time: # Time embedding
                dt = seq_dts[i]
                h_rad = 2.0 * math.pi * dt.hour / 24.0
                d_rad = 2.0 * math.pi * dt.dayofyear / 366.0
                time_dict = {"h_s": math.sin(h_rad), "h_c": math.cos(h_rad), "d_s": math.sin(d_rad), "d_c": math.cos(d_rad)}
                time_vec = torch.tensor([time_dict[k] for k in self.active_time], dtype=val_flat.dtype).expand(800, -1)
                val_flat = torch.cat([val_flat, time_vec], dim=-1)
                
            if self.active_pos: # Position embedding
                val_flat = torch.cat([val_flat, self.pos_emd], dim=-1)
            era5_frames_val.append(val_flat)
            
            # --------------- Historical GT ---------------
            hgt_val = self.nan_y[:, self.gt_keep_idx]
            if ts_str in self.gt_ts_map: 
                hgt_val = self.y_target_std[self.gt_ts_map[ts_str]][:, self.gt_keep_idx]
            hgt_frames_val.append(hgt_val)

        # 4. Stacking and padding
        gems_val = pad_sequence(frames_val, batch_first=True, padding_value=float("nan")) # (SEQ_LEN, max_N, C)
        gems_geo = pad_sequence(frames_geo, batch_first=True, padding_value=float("nan")) # (SEQ_LEN, max_N, 2)
        era5_val = torch.stack(era5_frames_val)    # (SEQ_LEN, 800, C)
        era5_geo = self.era5_geo_flat              # (800, 2)
        hgt_val = torch.stack(hgt_frames_val)      # (SEQ_LEN, N_stations, C_targets)
        hgt_geo = self.y_geo                       # (N_stations, 2)

        return gems_val, gems_geo, era5_val, era5_geo, hgt_val, hgt_geo, y_val, self.y_geo, uv_data

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file_handles'] = {} # Clear handles during pickling
        return state

def sparse_collate(batch):
    g_vals, g_geos, e_vals, e_geos, hgt_vals, hgt_geos, y_vals, y_geos, uv_data = zip(*batch)

    # Only GEMS has variable spatial points (N) across the batch
    max_n = max(x.shape[1] for x in g_vals)
    if g_vals[0].shape[2] == 0:
        g_vals = torch.stack([torch.zeros(x.shape[0], max_n, 0, dtype=x.dtype, device=x.device) for x in g_vals])
    else:
        g_vals = torch.stack([F.pad(x, (0, 0, 0, max_n - x.shape[1]), value=float('nan')) for x in g_vals])
    g_geos = torch.stack([F.pad(x, (0, 0, 0, max_n - x.shape[1]), value=float('nan')) for x in g_geos])
    era5_val, era5_geo = torch.stack(e_vals), e_geos[0].unsqueeze(0)
    hgt_val, hgt_geo = torch.stack(hgt_vals), hgt_geos[0].unsqueeze(0)
    y_val, y_geo = torch.stack(y_vals), y_geos[0].unsqueeze(0)
    return g_vals, g_geos, era5_val, era5_geo, hgt_val, hgt_geo, y_val, y_geo, torch.stack(uv_data)

def split_by_day(dataset, val_ratio=0.1, seed=42):
    day_to_indices = defaultdict(list) # Group indices by day (YYYYMMDD)
    for i, sample_idx in enumerate(dataset.sample_indices):
        ts_str = dataset.timestamps[sample_idx]
        dt = pd.to_datetime(ts_str, format="%Y%m%d%H%M") + pd.Timedelta(hours=8)
        day_str = dt.strftime("%Y%m%d")
        day_to_indices[day_str].append(i)

    unique_days = list(day_to_indices.keys())
    random.seed(seed)
    random.shuffle(unique_days)
    val_days_count = max(int(len(unique_days) * val_ratio), 1)
    val_days = unique_days[:val_days_count]
    train_days = unique_days[val_days_count:]

    # Flatten indices for Subset
    train_idx = [idx for day in train_days for idx in day_to_indices[day]]
    valid_idx = [idx for day in val_days for idx in day_to_indices[day]]

    return Subset(dataset, train_idx), Subset(dataset, valid_idx)
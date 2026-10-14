import _env
import torch, h5py, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils import SpatialProjector
from configs import cfg

def print_stat_row(name: str, v_min: float, v_max: float, v_mean: float, v_std: float, comp: float, label: str = "Land Comp"):
    """ Helper to print uniform feature summary rows. """
    print(f"{name:15s} | Min: {v_min:.4e} | Max: {v_max:.4e} | Mean: {v_mean:.4e} | Std: {v_std:.4e} | {label}: {comp:.2f}%")

def compute_land_feature_statistics():
    """ Computes raw statistics (Min, Max, Mean, Std) and land spatial completeness (%) for active features. """
    req = [cfg.STD_PATH, cfg.ERA5_PATH, cfg.GEMS_PATH, cfg.GT_PATH, cfg.STATIC_PATH]
    if missing := [f for f in req if not os.path.exists(f)]:
        print(f"[Skip] Missing required files: {missing}")
        return
    with h5py.File(cfg.STD_PATH, 'r') as f:
        g_min, g_max, g_mean, g_std = f['GEMS']['min'][:], f['GEMS']['max'][:], f['GEMS']['mean'][:], f['GEMS']['std'][:]
        e_min, e_max, e_mean, e_std = f['ERA5']['min'][:], f['ERA5']['max'][:], f['ERA5']['mean'][:], f['ERA5']['std'][:]
        gt_min, gt_max, gt_mean, gt_std = f['GT']['min'][:], f['GT']['max'][:], f['GT']['mean'][:], f['GT']['std'][:]

    # Extract ERA5 Land Mask (308 points)
    with h5py.File(cfg.ERA5_PATH, 'r') as f:
        era5_vals = f['val'][:] # (T, C, 40, 20)
        land_mask = ~np.isnan(era5_vals[0, 0])
        total_land_grid = np.sum(land_mask)
        total_era5_pts = len(era5_vals) * total_land_grid

    # GEMS Land Completeness (GPU Vectorized)
    active_gems = [name for name in cfg.GEMS_ABBRE if name in cfg.FEATURES]
    active_g_idx = [cfg.GEMS_ABBRE.index(name) for name in active_gems]
    projector = SpatialProjector().to(cfg.DEVICE)
    land_mask_t = torch.from_numpy(land_mask).to(cfg.DEVICE)
    accum_valid = {name: 0 for name in active_gems}

    with h5py.File(cfg.GEMS_PATH, 'r') as f:
        lens = f['len'][:]
        day_indices = np.where(lens > 0)[0]
        # Stream per timestep to avoid loading entire GEMS dataset into RAM
        for t in day_indices:
            n = int(lens[t])
            if n == 0: continue
            v = torch.from_numpy(f['val'][t:t+1, :n, active_g_idx]).unsqueeze(0).to(cfg.DEVICE)
            g = torch.from_numpy(f['geo'][t:t+1, :n, :]).unsqueeze(0).to(cfg.DEVICE)
            _, mask = projector.project_tensor(v, g, param=(40, 20, 0.1))
            valid_pts = ((mask[0, 0] > 0) & land_mask_t).sum(dim=(1, 2)).cpu().numpy()
            for name, pts in zip(active_gems, valid_pts):
                accum_valid[name] += pts

    print("=== 1. GEMS Satellite Features (Land Area Completeness vs 308 pts) ===")
    for name in active_gems:
        i = cfg.GEMS_ABBRE.index(name)
        comp = (accum_valid[name] / (len(day_indices) * total_land_grid)) * 100.0
        print_stat_row(name, g_min[i], g_max[i], g_mean[i], g_std[i], comp, "Land Comp")

    print("\n=== 2. ERA5 Weather Features (Land Area Completeness vs 308 pts) ===")
    active_era5 = [name for name in cfg.ERA5_ABBRE if name in cfg.FEATURES]
    for name in active_era5:
        i = cfg.ERA5_ABBRE.index(name)
        comp = ((~np.isnan(era5_vals[:, i])) & land_mask).sum() / total_era5_pts * 100.0
        print_stat_row(name, e_min[i], e_max[i], e_mean[i], e_std[i], comp, "Land Comp")

    print("\n=== 3. Ground Station (GT) Features ===")
    with h5py.File(cfg.GT_PATH, 'r') as f:
        gt_data = f['data'][:]
        total_gt_pts = gt_data.shape[0] * gt_data.shape[1]
        for i, name in enumerate(cfg.TARGETFEATURES):
            comp = (~np.isnan(gt_data[..., i])).sum() / total_gt_pts * 100.0
            print_stat_row(name, gt_min[i], gt_max[i], gt_mean[i], gt_std[i], comp, "Station Comp")

    print("\n=== 4. Static Topography Features ===")
    with h5py.File(cfg.STATIC_PATH, 'r') as f:
        val, mask = f['val'][:], f['mask'][:]
        for name in cfg.STATIC_ABBRE:
            if name in cfg.FEATURES:
                i = cfg.STATIC_ABBRE.index(name)
                v = val[i][mask[i] >= 0.5]
                print_stat_row(name, v.min(), v.max(), v.mean(), v.std(), 100.0, "Spatial Comp")

def analyze_external_pollutants_statistics(
    int_path: str = "./dataset_storage/processed/sparse_targets_2021_2025.h5",
    ext_path: str = "./dataset_storage/processed/external_2025.h5"
):
    """ Compares pollutant statistics across Train (75 sites), Test (75 sites), and External (26 sites). """
    if not os.path.exists(int_path):
        print(f"{int_path} does not exist !")
        return
    
    with h5py.File(int_path, "r") as f_int, h5py.File(ext_path, "r") as f_ext:
        data_int, ts_int = f_int["data"][:], f_int["timestamps"][:].astype("S").astype("U")
        cols_int = [c.decode("utf-8") if isinstance(c, bytes) else c for c in f_int["columns"]]
        data_ext = f_ext["data"][:]
        cols_ext = [c.decode("utf-8") if isinstance(c, bytes) else c for c in f_ext["columns"]]

    train_m = (ts_int >= "2021") & (ts_int < "2025")
    test_m = ts_int >= "2025"
    units = {"PM2.5": "ug/m3", "PM10": "ug/m3", "O3": "ppb", "NO2": "ppb", "CO": "ppm", "SO2": "ppb"}
    records = []

    print("\n" + "="*105)
    print("=== THREE-WAY COMPARATIVE STATISTICS: TRAIN (75 SITES) vs TEST (75 SITES) vs EXTERNAL (26 SITES) ===")
    print("="*105)

    for idx_int, feat in enumerate(cols_int):
        idx_ext = cols_ext.index(feat)
        v_tr = data_int[train_m, :, idx_int][~np.isnan(data_int[train_m, :, idx_int])]
        v_te = data_int[test_m, :, idx_int][~np.isnan(data_int[test_m, :, idx_int])]
        v_ext = data_ext[..., idx_ext][~np.isnan(data_ext[..., idx_ext])]

        m_tr = float(np.mean(v_tr)) if len(v_tr) else np.nan
        m_te = float(np.mean(v_te)) if len(v_te) else np.nan
        m_ext = float(np.mean(v_ext)) if len(v_ext) else np.nan

        p50_tr = np.percentile(v_tr, 50) if len(v_tr) else np.nan
        p50_te = np.percentile(v_te, 50) if len(v_te) else np.nan
        p50_ext = np.percentile(v_ext, 50) if len(v_ext) else np.nan

        spatial_shift = ((m_ext - m_te) / m_te) * 100.0 if not np.isnan(m_te) and m_te != 0 else np.nan

        records.append({
            "Pollutant": f"{feat} ({units.get(feat, '')})",
            "Train 21-24 (75 sites)": f"{m_tr:.2f} (P50: {p50_tr:.2f})",
            "Test 2025 (75 sites)": f"{m_te:.2f} (P50: {p50_te:.2f})",
            "Ext 2025 (26 sites)": f"{m_ext:.2f} (P50: {p50_ext:.2f})" if not np.isnan(m_ext) else "No Data (All NaN)",
            "Ext Range [Min, Max]": f"[{np.min(v_ext):.1f}, {np.max(v_ext):.1f}]" if len(v_ext) else "[NaN, NaN]",
            "2025 Spatial Shift": f"{spatial_shift:+.1f}%" if not np.isnan(spatial_shift) else "N/A"
        })

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    print("="*105 + "\n")

def view_sparse_feature(target_ts: str, abbre: str):
    """ Directly visualizes raw GEMS spatial distribution for a target timestamp. """
    if abbre not in cfg.ABBRE_TRANS:
        print(f"[Warn] Feature abbreviation '{abbre}' not in cfg.ABBRE_TRANS.")
        return

    # Convert query hourly timestamp to GEMS scan timestamp (-15m)
    raw_ts = (pd.to_datetime(target_ts, format="%Y%m%d%H%M") - pd.Timedelta(minutes=15)).strftime("%Y%m%d%H%M")
    f_name = cfg.ABBRE_TRANS[abbre]

    with h5py.File(cfg.GEMS_PATH, 'r') as hf:
        ts_arr = hf['timestamps'][:].astype('S').astype('U')
        match = np.where(ts_arr == raw_ts)[0]
        if len(match) == 0:
            print(f"[Warn] Target timestamp '{target_ts}' (raw: '{raw_ts}') not found in GEMS dataset.")
            return
        idx = int(match[0])
        n = int(hf['len'][idx])
        c_idx = [ch.decode('utf-8') if isinstance(ch, bytes) else ch for ch in hf.attrs['channels']].index(f_name)
        vals, geos = hf['val'][idx, :n, c_idx], hf['geo'][idx, :n]

    valid = ~np.isnan(vals)
    if not np.any(valid):
        print(f"[Warn] No valid data for {abbre} at {target_ts}.")
        return

    _, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    sc = ax.scatter(geos[valid, 0], geos[valid, 1], c=vals[valid], cmap='turbo', s=6, zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=2)
    ax.gridlines(draw_labels=["left", "bottom"], linestyle='--', alpha=0.5)
    ax.set_extent([*cfg.GEO_AREA[0], *cfg.GEO_AREA[1]], crs=ccrs.PlateCarree())
    plt.colorbar(sc, ax=ax, label=abbre, shrink=0.8, pad=0.03)
    ax.set_title(f"{abbre} (Raw Data) - {target_ts}", fontsize=12, fontweight='bold')
    
    save_path = f"sparse_{abbre}_{target_ts}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> Saved: {save_path} (Valid points: {np.sum(valid)})")

if __name__ == "__main__":
    compute_land_feature_statistics()
    analyze_external_pollutants_statistics()
    view_sparse_feature("202501020700", "AOD")

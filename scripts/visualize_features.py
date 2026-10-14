import _env
import torch, cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Disable GUI window popup
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch.utils.data import DataLoader
from pipelines import batch_process
from data import SparseDataGenerator, sparse_collate
from data.dataset import GEMS_FEA, ERA5_ABBRE
from utils import SpatialProjector
from configs import cfg
from tqdm import tqdm

# === Core Settings: Freely switchable between "GEMS" and "ERA5" ===
DATA_TYPE = "ERA5" 

if DATA_TYPE == "GEMS":
    FEATURES_TO_PLOT = [f for f in cfg.FEATURES if f in cfg.GEMS_ABBRE]
    ALL_CHANNELS = GEMS_FEA
    STATS_KEY = "GEMS"
else:
    FEATURES_TO_PLOT = [f for f in cfg.FEATURES if f in cfg.ERA5_ABBRE]
    ALL_CHANNELS = ERA5_ABBRE
    STATS_KEY = "ERA5"

# Dynamically calculate the number of sub-picture layout rows and columns
N_FEAS = len(FEATURES_TO_PLOT)
COLS = 3 if N_FEAS >= 3 else N_FEAS
ROWS = (N_FEAS + COLS - 1) // COLS

lon = np.arange(120.0, 122.0, cfg.RESOLU)
lat = np.arange(21.5, 25.5, cfg.RESOLU)
lon_mesh, lat_mesh = np.meshgrid(lon, lat)

def get_raw_subplots_array(gems_val, gems_geo, title_suffix):
    fig, axes = plt.subplots(ROWS, COLS, figsize=(9, 16), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten() if ROWS * COLS > 1 else np.array([axes])
    
    gems_val_unnorm = gems_val[0]
    geos = gems_geo[0, -1].numpy() if gems_geo.ndim == 4 else (gems_geo[0].numpy() if gems_geo.ndim == 3 else gems_geo.numpy())
    
    limits = {}
    for ax, fea in zip(axes, FEATURES_TO_PLOT):
        full_name = cfg.ABBRE_TRANS.get(fea, fea)
        f_idx = ALL_CHANNELS.index(full_name)
        vals = gems_val_unnorm[-1, :, f_idx].numpy()
        
        valid = ~np.isnan(vals)
        vals_filtered, geos_filtered = vals[valid], geos[valid]
        
        if len(vals_filtered) > 0:
            v_min, v_max = np.nanmin(vals_filtered), np.nanmax(vals_filtered)
            if v_min == v_max:
                v_min, v_max = v_min - 1.0, v_max + 1.0
            limits[fea] = (v_min, v_max)
            
            sc = ax.scatter(geos_filtered[:, 0], geos_filtered[:, 1], c=vals_filtered, cmap='jet', s=8, 
                            vmin=v_min, vmax=v_max, transform=ccrs.PlateCarree(), zorder=3, edgecolors='none')
            fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.8, aspect=30, pad=0.01)
            
        ax.set_extent([cfg.GEO_AREA[0][0], cfg.GEO_AREA[0][1], cfg.GEO_AREA[1][0], cfg.GEO_AREA[1][1]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=2)
        ax.set_title(f"{fea} (Scatter)", fontsize=16)
        
    plt.suptitle(f"Feature Distribution ({title_suffix})", fontsize=22, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.02, hspace=0.10, left=0.01, right=0.99, bottom=0.01, top=0.93)
    
    for ax in axes[len(FEATURES_TO_PLOT):]:
        ax.axis('off')
        
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img, limits

def get_interp_subplots_array(x_grid, limits, title_suffix):
    fig, axes = plt.subplots(ROWS, COLS, figsize=(9, 16), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten() if ROWS * COLS > 1 else np.array([axes])
    
    for ax, fea in zip(axes, FEATURES_TO_PLOT):
        grid_idx = cfg.FEATURES.index(fea)
        grid_data = x_grid[0, -1, grid_idx, :, :].cpu().numpy()
        v_min, v_max = limits.get(fea, (np.nanmin(grid_data), np.nanmax(grid_data)))
            
        mesh = ax.pcolormesh(lon_mesh, lat_mesh, grid_data,
                             vmin=v_min, vmax=v_max, cmap='jet', transform=ccrs.PlateCarree(), zorder=1)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=2)
        ax.set_extent([cfg.GEO_AREA[0][0], cfg.GEO_AREA[0][1], cfg.GEO_AREA[1][0], cfg.GEO_AREA[1][1]], crs=ccrs.PlateCarree())
        ax.set_title(f"{fea} (Grid)", fontsize=16)
        fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, aspect=30, pad=0.01)
        
    plt.suptitle(f"Feature Distribution ({title_suffix})", fontsize=22, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.02, hspace=0.10, left=0.01, right=0.99, bottom=0.01, top=0.93)
    
    for ax in axes[len(FEATURES_TO_PLOT):]:
        ax.axis('off')
        
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True

    # Initialize dataset restricted to Jan 2025
    projector = SpatialProjector().to(cfg.DEVICE)
    common_paths = [cfg.GEMS_PATH, cfg.ERA5_PATH, cfg.GT_PATH]
    jan_set = SparseDataGenerator(*common_paths, "20250101", "20250111", cfg.STD_PATH)
    train_loader = DataLoader(jan_set, 1, False, collate_fn=sparse_collate, num_workers=2)
    print(f"Total timestamps in Jan 2025: {len(jan_set)}")

    video_out = "./results/raw_vs_interpolation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = None

    for i, batch in enumerate(tqdm(train_loader, desc="Generating Video")):
        ts_str = jan_set.timestamps[jan_set.sample_indices[i]]
        gems_val, gems_geo, era5_val, era5_geo = batch[:4]
        
        # Skip timestamps with insufficient valid GEMS observations
        if DATA_TYPE == "GEMS":
            valid_points = torch.sum(~torch.isnan(gems_val[0, -1, :, 0])).item()
            if valid_points < 10:
                continue
            
        val_batch = gems_val if DATA_TYPE == "GEMS" else era5_val
        geo_batch = gems_geo if DATA_TYPE == "GEMS" else era5_geo
        
        # ------------------ Raw (Scatter Plot) ------------------
        img_raw, limits = get_raw_subplots_array(val_batch, geo_batch, f"{ts_str} - Raw (Scatter)")
        
        # ------------------ Grid (pcolormesh) ------------------
        projector.interpolation = True
        pb_interp = batch_process(batch, projector, cfg.to_device)
        img_interp = get_interp_subplots_array(pb_interp.x_grid, limits, f"{ts_str} - Interpolated (Grid)")
        
        # ------------------ Horizontal Concatenation ------------------
        h_interp, w_interp, _ = img_interp.shape
        img_raw_resized = cv2.resize(img_raw, (w_interp, h_interp))
        combined = np.hstack((img_raw_resized, img_interp))
        frame = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        if video is None:
            h_frame, w_frame, _ = frame.shape
            video = cv2.VideoWriter(video_out, cv2.CAP_MSMF, fourcc, 3, (w_frame, h_frame))
            if not video.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for output: {video_out}")

        video.write(frame)
        del pb_interp

    if video is not None:
        video.release()
        print("Done! Video successfully generated.")

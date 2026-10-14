from rasterio.transform import from_origin
from configs import cfg
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import rasterio, os, matplotlib, calendar
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_on_map(grid_data, resolution=0.005, output_dir="", file_name="ground true", mute=True):
    """
    Visualizes 6-channel grid data on a map with coastlines and saves it.
    Args:
        grid_data (np.ndarray): The data to plot, shape=(6, 800, 400).
        output_dir (str): Directory to save the output images.
        file_prefix (str): Prefix for the output filename (e.g., 'prediction' or 'ground_truth').
    """
    # Fetch physical domain configurations directly from config
    pollutants = list(cfg.POLLUTANT_DOMAINS.keys())
    
    # Create figure and axes with Cartopy projection
    fig, axes = plt.subplots(2, 3, figsize=(11, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    lon, lat = np.arange(120.0, 122.0, resolution), np.arange(21.5, 25.5, resolution)

    # Get coordinate grids and define high-resolution ocean layer to mask out sea areas
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    ocean_10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='white', edgecolor='none')

    for i in range(6):
        ax = axes[i]
        pollutant_name = pollutants[i]
        vmin, vmax = cfg.POLLUTANT_DOMAINS[pollutant_name]
        # Use pcolormesh for plotting
        mesh = ax.pcolormesh(lon_mesh, lat_mesh, grid_data[i, :, :],
                               vmin=vmin, vmax=vmax, cmap='jet', transform=ccrs.PlateCarree(), zorder=1)
        
        # Overlay white ocean cover to hide ocean values, and draw coastline/borders on top
        ax.add_feature(ocean_10m, zorder=1.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=2)
        
        # Add gridlines and labels
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set title and colorbar
        ax.set_title(pollutant_name)
        fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=1, aspect=35, pad=0.01)

    plt.suptitle(f'{file_name.replace("_", " ")} Distribution', fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.93, wspace=0.01, hspace=0.12)

    # Save the figure
    plt.savefig(output_path := os.path.join(output_dir, f"{file_name}.png"), dpi=300, bbox_inches='tight')
    if not mute:    print(f"Saved map to {output_path}")
    plt.close(fig)

def plot_loss(loss, loss_type, mode, outpath=""):
    loss_array = np.array(loss)
    x = np.arange(1, len(loss_array) + 1) # Vectorized generation
    train_loss, val_loss = loss_array[:, 0], loss_array[:, 1]
    fig = plt.figure()
    plt.plot(x, train_loss, 'r-', label='Train Loss')
    plt.plot(x, val_loss, 'b-', label='Validation Loss')
    if np.nanmax(loss_array) / np.nanmin(loss_array[loss_array > 0]) > 100:
        plt.yscale('log')
    plt.xlabel("Epochs", fontsize=14, fontname='Arial')
    plt.ylabel(loss_type, fontsize=14, fontname='Arial')
    plt.title(f"{mode} Loss", fontsize=18, fontfamily='serif')
    plt.legend()
    plt.savefig(os.path.join(outpath, f"{mode}_{loss_type}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_hourly_error(feature, mae, rmse, r2, mape, t_length=6, outpath="./results/", model="AttentionUnet"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{feature} Error on Different Timestamps ({model})', fontsize=16, fontweight='bold')
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C']
    x = np.arange(t_length)
    
    # Define metrics and their corresponding axes
    metrics = [
        ('MAE', mae, axes[0, 0]),
        ('RMSE', rmse, axes[0, 1]),
        ('R square', r2, axes[1, 0]),
        ('MAPE', mape, axes[1, 1])
    ]
    
    for i, (title, data, ax) in enumerate(metrics):
        ax.bar(x, data, width=0.4, color=colors[i], edgecolor='black', zorder=3)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"t{i+1}" for i in range(t_length)])
        ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path := os.path.join(outpath, f"{model}_hourly_error.png"), dpi=300)
    print(f"Saved plot: {save_path}")
    plt.close(fig)

def save_as_geotiff(grid_data, output_path, resolution=0.005):
    """
    Save (Channels, Height, Width) Numpy Array as GeoTIFF
    Args:
        grid_data: (C=6, H=800, W=400)
        output_path: filename.tif
        resolution: grid resolution
    """
    # get shape (Channel, Height, Width)
    if grid_data.ndim == 2:    grid_data = grid_data[np.newaxis, :, :] # Add Channel dimension
    count, height, width = grid_data.shape

    # Define coordinates (North-West is the Origin of GeoTIFF)
    west, north = 120.0, 25.5
    
    # Create Affine Transform
    transform = from_origin(west, north, resolution, resolution)

    #  *** Flip Up-Down ***
    data_to_save = np.flip(grid_data, axis=1) 

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,               # band number (6)
        dtype=data_to_save.dtype,  # float32 or float64
        crs='EPSG:4326',           # Coordinates system WGS84
        transform=transform,
        nodata=np.nan,             # filled with NaN
        compress='lzw'             # compression format
    ) as dst:
        dst.write(data_to_save)
        
        # Setting band names
        band_names = cfg.TARGETFEATURES
        if count == len(band_names):
            dst.descriptions = band_names

    print(f"GeoTIFF saved to: {output_path}")

def plot_monthly_error(feature, mae, rmse, r2, mape, months, outpath="./results/", model="AttentionUnet"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{feature} Monthly Performance ({model})', fontsize=16, fontweight='bold')
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C']
    x = np.arange(len(months))
    
    # Define metrics and their corresponding axes
    metrics = [
        ('MAE', mae, axes[0, 0]),
        ('RMSE', rmse, axes[0, 1]),
        ('R square', r2, axes[1, 0]),
        ('MAPE', mape, axes[1, 1])
    ]
    
    for i, (title, data, ax) in enumerate(metrics):
        ax.bar(x, data, width=0.5, color=colors[i], edgecolor='black', zorder=3)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([calendar.month_abbr[m] for m in months])
        ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
        
    os.makedirs(outpath, exist_ok=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(outpath, f"{model}_monthly_error.png"), dpi=300)
    plt.close(fig)

def plot_type_metrics(feature, type_names, metrics_storage, mode="sitetype", output_dir="", model="AttentionUnet"):
    """
    Visualizes metrics across different types (sitetype or area).
    metrics_storage structure: {type_names: {feature: [R2, MAE, RMSE, MAPE]}}
    """
    labels = ['R square', 'MAE', 'RMSE', 'MAPE']
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C'] # Anime style palette (Vibrant Pastel)
    x = np.arange(len(type_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot 4 bars per site
    offsets = np.linspace(-1.5 * width, 1.5 * width, 4)
    
    for i, metric_name in enumerate(labels):
        # Extract the i-th metric for the specific feature across all sites
        values = [metrics_storage[tp][feature][i] for tp in type_names]
        if i == 3:
            values = [v/100 for v in values] # MAPE
        ax.bar(x + offsets[i], values, width, label=metric_name, 
               color=colors[i], edgecolor='black', linewidth=1.2, zorder=3)

    ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
    ax.set_title(f'{feature} Performance by {mode}', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(type_names, fontsize=11, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, framealpha=1, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    plt.tight_layout()
    plt.savefig(save_path := os.path.join(output_dir, f"{model}_{mode}_error.png"), dpi=300)
    print(f"Saved plot: {save_path}")
    plt.close(fig)

def plot_distribution(pred, gt, mask, names, bins=50, save_path="./results/", filename="distribution"):
    # iterate over 6 pollutant channels
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ['#FF6B6B', '#4ECDC4']
    for i, ax in enumerate(axes.flatten()):
        vp, vg = pred[:, i, ...][mask[:, i, ...]], gt[:, i, ...][mask[:, i, ...]]

        # Avoid crash if mask covers all area
        if vp.size == 0 or vg.size == 0:
            ax.set_title(f"{names[i]} (No Data)")
            continue
        
        # determine local range for each pollutant
        min_v, max_v = min(vp.min(), vg.min()).item(), max(vp.max(), vg.max()).item()
        ax.hist(vg, bins=bins, range=(min_v, max_v), alpha=0.6, label='GT', color=colors[0])
        ax.hist(vp, bins=bins, range=(min_v, max_v), alpha=0.6, label='Pred', color=colors[1])
        ax.set_title(names[i], fontsize=16, fontweight='bold')
        if i>=3:
            ax.set_xlabel('Value', fontsize=14, fontweight='bold')
        if i%3==0:
            ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.legend()
        
    plt.subplots_adjust(left=0.06, right=0.95, bottom=0.05, top=0.93, wspace=0.17, hspace=0.14)
    plt.savefig(os.path.join(save_path, f"{filename}.png"), dpi=300)
    plt.close(fig)

def plot_mad(feature, type_names, metrics_storage, mode="sitetype", output_dir="./results/outliers"):
    """
    Visualizes extreme thresholds across different types (sitetype or area).
    metrics_storage structure: {type_names: {feature: threshold_value}}
    """
    # Anime style palette (Vibrant Pastel)
    color = '#FF6B6B' 
    x = np.arange(len(type_names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract values for the specific feature
    values = [metrics_storage[tp].get(feature, 0) for tp in type_names]
    
    ax.bar(x, values, width, label='MAD (Median Absolute Deviation)', 
           color=color, edgecolor='black', linewidth=1.2, zorder=3)

    ax.set_ylabel('Threshold Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{feature} Extreme Threshold by {mode}', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(type_names, fontsize=11, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, framealpha=1, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    plt.tight_layout()
    plt.savefig(save_path := os.path.join(output_dir, f"{feature}_{mode}_outlier.png"), dpi=300)
    print(f"Saved plot: {save_path}")
    plt.close(fig)

def plot_monthly_mad(thresholds, outpath="./results/outliers"):
    os.makedirs(outpath, exist_ok=True)
    months = list(range(1, 13))
    month_labels = [calendar.month_abbr[m] for m in months]
    
    # Plot line chart for each feature
    for fea, vals in thresholds.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(months, vals, marker='o', linestyle='-', color='#FF6B6B', linewidth=2, markersize=6)
        ax.set_title(f'Monthly Extreme Threshold: {fea}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Threshold', fontsize=12, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels, fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f'mad_threshold_{fea}.png'), dpi=300)
        plt.close(fig)
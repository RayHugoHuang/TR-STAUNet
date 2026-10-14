import os, torch
import numpy as np
from dataclasses import dataclass
from utils import Evaluator, create_masks, plot_type_metrics, plot_mad, get_monthly_mad
SITE_CSV_PATH = "./dataset_storage/site_coordinate.csv"

@dataclass
class ProjectedBatch:
    x_grid: torch.Tensor
    x_mask: torch.Tensor = None
    y_grid: torch.Tensor = None
    y_mask: torch.Tensor = None
    uv_grid: torch.Tensor = None
    uv_mask: torch.Tensor = None

@torch.no_grad()
def batch_process(batch, projector, to_device, aug=None, shift=40, drop=0.1, project_uv=False):
    # Unpack the batch (9 items)
    gems_val, gems_geo, era5_val, era5_geo, hgt_val, hgt_geo, y_val, y_geo, uv_data = batch
    
    # Send only the required tensors to device
    gems_val, gems_geo, era5_val, era5_geo, hgt_val, hgt_geo, y_val, y_geo = map(
        to_device, (gems_val, gems_geo, era5_val, era5_geo, hgt_val, hgt_geo, y_val, y_geo)
    )

    x_grid, x_mask = projector(gems_val, gems_geo, era5_val, era5_geo, hgt_val, hgt_geo)
    
    T = y_val.shape[1]
    # Target is the single frame at t+1~T, Shape: (B, T, C_out, H, W) fp32
    y_grid, y_mask = projector.project_gt(y_val, y_geo)

    if project_uv: # u, v, dem at t+1~T (for Physical loss)
        uv_grid, uv_mask = projector.project_gt(to_device(uv_data), era5_geo, interpolation=True)
        uv_grid = torch.cat([uv_grid, x_grid[:, -1:, -1:, :, :].expand(-1, T, -1, -1, -1)], dim=2)
        uv_mask = torch.cat([uv_mask, x_mask[:, -1:, -1:, :, :].expand(-1, T, -1, -1, -1)], dim=2)
    else:
        uv_grid, uv_mask = None, None
    
    if aug is not None:
        x_grid, x_mask, y_grid, y_mask = aug.do(x_grid, x_mask, y_grid, y_mask, shift, drop)
        
    # Return x_mask as None to immediately free its large memory (not used after batch_process)
    return ProjectedBatch(x_grid, None, y_grid, y_mask, uv_grid, uv_mask)

def evaluate_and_log(features, Y_True, Y_Pred, Y_Mask, timestep, model_name, log_path, output_dir, header_msg, do_plot_mad=False):
    for fea in features:    os.makedirs(os.path.join(output_dir, fea), exist_ok=True)
    eva = Evaluator()
    with open(log_path, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(str(msg) + "\n")
            
        log(header_msg)
        eva.all_errors(features, Y_True, Y_Pred, Y_Mask, log_file=f)
        if timestep:
            eva.hourly_error(features, timestep, Y_True, Y_Pred, Y_Mask, model_name, output_dir)
            eva.monthly_error(features, timestep, Y_True, Y_Pred, Y_Mask, model_name, output_dir)

        taipei_mask, others_mask = create_masks(target_country="臺北市", csv_path=SITE_CSV_PATH)
        nantou_mask, _ = create_masks(target_country="南投縣", csv_path=SITE_CSV_PATH)
        for label, mask in [("Taipei", taipei_mask), ("other cities", others_mask), ("Nantou", nantou_mask)]:
            log(f"\n------ Aggregate Metrics in {label} ------")
            eva.all_errors(features, Y_True, Y_Pred, Y_Mask, log_file=f, spatial_mask=mask)
        
        # Sitetype & Area evaluation
        for group_col, mode_str in [("sitetype", "sitetype"), ("areaname", "area")]:
            storage, mad_storage = {}, {}
            masks, _ = create_masks(group_col=group_col, csv_path=SITE_CSV_PATH)
            valid_names = [name for name in masks.keys() if name != "其他"]
            
            for name in valid_names:
                log(f"\n------ Aggregate Metrics in {name} ------")
                sp_mask = masks[name]
                storage[name] = eva.all_errors(features, Y_True, Y_Pred, Y_Mask, "", f, spatial_mask=sp_mask)
                # Calculate MAD threshold if enabled
                if do_plot_mad:
                    mad_storage[name] = {}
                    for i, fea in enumerate(features):
                        feature_mask = Y_Mask[:, i] & sp_mask
                        valid_data = Y_True[:, i][feature_mask]
                        if len(valid_data) > 0:
                            median = np.median(valid_data)
                            mad = np.median(np.abs(valid_data - median))
                            mad_storage[name][fea] = median + 3 * (1.4826 * mad) # Outlier threshold
                        else:
                            mad_storage[name][fea] = 0
            
            for fea in features:
                out_p = os.path.join(output_dir, fea)
                plot_type_metrics(fea, storage.keys(), storage, mode=mode_str, output_dir=out_p, model=model_name)
                if do_plot_mad:
                    plot_mad(fea, valid_names, mad_storage, mode=mode_str)
            
        if do_plot_mad:    get_monthly_mad(features, timestep, Y_True, Y_Mask, output_dir)
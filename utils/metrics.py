import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from .visualization import plot_hourly_error, plot_monthly_error, plot_monthly_mad

class Evaluator:
    """A collection of static methods for evaluating model predictions."""
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_mask: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calculates MAE, RMSE, R-squared, and MAPE for given true and predicted values.
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            y_mask: A mask array to indicate which values to include in the calculation.
        Returns:
            A tuple containing MAE, RMSE, R-squared, and MAPE.
        """
        # Convert mask to boolean and extract valid elements directly
        mask = y_mask.astype(bool)
        if not np.any(mask):    return 0.0, 0.0, 0.0, 0.0
        yt, yp = y_true[mask], y_pred[mask]

        # Calculate MAE and RMSE
        mae, rmse = np.mean(np.abs(yp - yt)), np.sqrt(np.mean(np.square(yp - yt)))

        # Calculate R square
        ss_tot, ss_res = np.sum(np.square(yt - np.mean(yt))), np.sum(np.square(yt - yp))
        if ss_tot < 1e-8:
            r2 = 1.0 if ss_res < 1e-8 else 0.0
        else:
            r2 = 1.0 - (ss_res / ss_tot)

        # Calculate MAPE (Avoid division by zero with boolean indexing)
        mape_mask = np.abs(yt) > 1e-5
        mape = 100.0 * np.mean(np.abs((yp[mape_mask] - yt[mape_mask]) / yt[mape_mask])) if np.any(mape_mask) else np.nan

        return np.round(mae, 2), np.round(rmse, 2), np.round(r2, 2), np.round(mape)

    @staticmethod
    def all_errors(target_features: List[str], y_true_all: np.ndarray, 
                   y_pred_all: np.ndarray, y_mask_all: np.ndarray, mode: str="mute", log_file=None, spatial_mask=None):
        """
        Calculates and prints aggregate error metrics for each target feature.
        Args:
            target_features: A list of feature names.
            y_true_all: A numpy array of all ground truth values with shape (N, C, H, W).
            y_pred_all: A numpy array of all predicted values with shape (N, C, H, W).
            y_mask_all: A numpy array of all mask values with shape (N, C, H, W).
        """
        metrics_storage = {fea:[] for fea in target_features}
        # 6 features (PM2.5, PM10, O3, NO2, CO, SO2)
        for i, fea in enumerate(target_features):
            # Extract all N data points (N, H, W) for this pollutant
            y_true, y_pred, y_mask = y_true_all[:, i, ...], y_pred_all[:, i, ...], y_mask_all[:, i, ...]
            if spatial_mask is not None:    y_mask = y_mask & spatial_mask
            mae, rmse, r2, mape = Evaluator.evaluate(y_true, y_pred, y_mask)
            
            metrics_storage[fea] = [r2, mae, rmse, mape]
            msg = f"{fea} R square: {r2:.2f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE(%): {mape}"
            if log_file is not None:    log_file.write(msg+"\n")
            if mode == "mute":    print(msg)

        if mode != "mute":    return metrics_storage

    def hourly_error(self, target_features, timestep, Y_True_All, Y_Pred_All, Y_Mask_All, model, outpath="./results/"):
        # Extract valid 6-hour sequence starting indices once
        times = pd.to_datetime(timestep, format='%Y%m%d%H%M')
        diffs = times.to_series().diff().dt.total_seconds().values / 3600
        start_indices = []
        for j in range(len(times) - 5):
            # Check if it's the start of a 6-hour chunk and contiguous
            if (j == 0 or diffs[j] != 1) and all(diffs[j+1:j+6] == 1):    start_indices.append(j)
        
        # Pre-calculate hour indices outside feature loop
        indices_per_hour = [[s + t for s in start_indices] for t in range(6)]
        for i, fea in enumerate(target_features):
            y_true, y_pred, y_mask = Y_True_All[:, i], Y_Pred_All[:, i], Y_Mask_All[:, i]
            # Aggregate and evaluate by hour offset (0 to 5)
            metrics = [self.evaluate(y_true[idx], y_pred[idx], y_mask[idx]) for idx in indices_per_hour]
            MAE, RMSE, R2, MAPE = zip(*metrics)
            MAPE = [m/100 for m in MAPE]
            
            plot_hourly_error(fea, MAE, RMSE, R2, MAPE, outpath=f"{outpath}/{fea}", model=model)

    def monthly_error(self, target_features, timestep, Y_True_All, Y_Pred_All, Y_Mask_All, model, outpath="./results/"):
        # Pre-calculate month array once to boost performance and boolean masks for each month
        months = pd.to_datetime(timestep, format='%Y%m%d%H%M').month.values
        m_indices = {m: (months == m) for m in range(1, 13)}
        for i, fea in enumerate(target_features):
            y_true, y_pred, y_mask = Y_True_All[:, i], Y_Pred_All[:, i], Y_Mask_All[:, i]
            valid_months = [m for m in range(1, 13) if np.any(m_indices[m])]
            
            if valid_months:
                metrics = [self.evaluate(y_true[m_indices[m]], y_pred[m_indices[m]], y_mask[m_indices[m]]) for m in valid_months]
                MAE, RMSE, R2, MAPE = zip(*metrics)
                MAPE = [m/100 for m in MAPE]
                plot_monthly_error(fea, MAE, RMSE, R2, MAPE, valid_months, outpath=f"{outpath}/{fea}/", model=model)

def create_masks(csv_path: str = "./dataset_storage/site_coordinate.csv", group_col: str = "county", 
                 target_country: Optional[str] = None, grid_params: dict = None):
    """
    Unified function to create masks for Types, Areas, or Regions.
    Args:
        target_val: If set, returns (target_mask, other_mask). If None, returns dict of all groups.
    """
    if group_col not in ['county', 'sitetype', 'areaname']:    raise ValueError
    df = pd.read_csv(csv_path)
    
    # Defaults or Custom params
    gp = grid_params or {'lat_min': 21.5, 'lon_min': 120.0, 'res': 0.005, 'H': 800, 'W': 400}
    H, W = gp['H'], gp['W']
    
    def build_grid(sub_df):
        mask = np.zeros((H, W), dtype=bool)
        if sub_df.empty: return mask
        y = np.round((sub_df['twd97lat'].values - gp['lat_min']) / gp['res']).astype(int)
        x = np.round((sub_df['twd97lon'].values - gp['lon_min']) / gp['res']).astype(int)
        valid = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        mask[y[valid], x[valid]] = True
        return mask

    if target_country:  # For create_region_masks logic
        mask_target = build_grid(df[df[group_col] == target_country])
        mask_others = build_grid(df[df[group_col] != target_country])
        return mask_target, mask_others
    else: # For create_type/area_masks logic
        return {group: build_grid(df[df[group_col] == group]) for group in df[group_col].unique()}, df[group_col].unique()

def get_monthly_mad(target_features, timestep, y_true, y_mask, outpath="./results/"):
    thresholds = {fea: np.zeros(12) for fea in target_features}
    # Extract month indices once to avoid redundant datetime parsing and pre-calculate boolean masks for each month
    months = pd.to_datetime(timestep, format='%Y%m%d%H%M').month.values - 1
    m_indices = {m: (months == m) for m in range(12)}

    for i, fea in enumerate(target_features):
        for month_idx in range(12):
            m_idx = m_indices[month_idx] # Find indices for the current month
            valid_yt = y_true[m_idx, i][y_mask[m_idx, i]] # Extract valid true values for specific feature and month
            
            # Calculate MAD threshold if valid data exists
            if valid_yt.size > 0:
                median = np.median(valid_yt)
                mad = np.median(np.abs(valid_yt - median))
                thresholds[fea][month_idx] = median + 3 * (1.4826 * mad)
            else:
                thresholds[fea][month_idx] = np.nan
    
    plot_monthly_mad(thresholds, outpath=f"{outpath}/")
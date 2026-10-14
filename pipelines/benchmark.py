import torch, os, shutil
import numpy as np
from tqdm import tqdm
from catboost import CatBoostRegressor
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from configs import cfg
from utils.visualization import plot_monthly_error, plot_on_map
from utils import SpatialProjector
TARGETFEATURES = cfg.TARGETFEATURES

class Trainer_CatBoost:
    def __init__(self):
        self.model = CatBoostRegressor(
            loss_function='MultiRMSE',
            iterations=1400,
            boosting_type = 'Plain',
            learning_rate=0.4,
            task_type="GPU" if torch.cuda.is_available() else "CPU",
            random_seed=42,
            verbose=False)
        self.projector = SpatialProjector(static_path="")
        self.projector.interpolation = True

    def batch_process(self, batch):
        g_vals, g_geos, e_vals, e_geos, hgt_val, hgt_geo, y_vals, y_geos, _ = batch
        B, N_GT = y_vals.shape[0], y_geos.shape[1]
        
        X_out, Y_out = [], []
        for b in range(B):
            target_geo = y_geos[0] # (N_GT, 2)
            
            # 1. Align ERA5
            if e_geos[0].shape[0] > 0:
                e_idx = torch.argmin(torch.cdist(target_geo, e_geos[0]), dim=1)
                e_feat = e_vals[b, :, e_idx, :].permute(1, 0, 2).reshape(N_GT, -1)
            else:
                e_feat = torch.full((N_GT, e_vals.shape[1] * e_vals.shape[3]), torch.nan, device=target_geo.device)
            
            # 2. Align GEMS
            if g_geos[b, 0].shape[0] > 0:
                g_idx = torch.argmin(torch.cdist(target_geo, g_geos[b, 0]), dim=1)
                g_feat = g_vals[b, :, g_idx, :].permute(1, 0, 2).reshape(N_GT, -1)
            else:
                g_feat = torch.full((N_GT, g_vals.shape[1] * g_vals.shape[3]), torch.nan, device=target_geo.device)
            
            # 3. Align HGT
            hgt_feat = hgt_val[b].permute(1, 0, 2).reshape(N_GT, -1)
            
            # 4. Combine & Filter
            x = torch.cat([g_feat, e_feat, hgt_feat, target_geo], dim=-1)
            y = y_vals[b]
            if y.dim() == 3: 
                y = y[0] # CatBoost only takes the first step (t=1) as a baseline comparison
            
            mask = ~torch.isnan(y).any(dim=1)
            if mask.any():
                X_out.append(x[mask])
                Y_out.append(y[mask])
                
        if len(X_out) == 0:
            return np.empty((0, x.shape[-1])), np.empty((0, 6))
        # (N, C)
        return np.vstack([t.cpu().numpy() for t in X_out]), np.vstack([t.cpu().numpy() for t in Y_out])

    def _dense_inference_and_plot(self, batch, timestamp):
        g_vals, g_geos, e_vals, e_geos, hgt_val, hgt_geo, _, _, _ = batch
        device = g_vals.device
        
        lat = torch.arange(21.5, 25.5, 0.005, device=device)
        lon = torch.arange(120.0, 122.0, 0.005, device=device)
        grid_lat, grid_lon = torch.meshgrid(lat, lon, indexing='ij')
        dense_geo = torch.stack([grid_lon.flatten(), grid_lat.flatten()], dim=-1)
        
        x_grid, _ = self.projector(g_vals, g_geos, e_vals, e_geos, hgt_val, hgt_geo)
        C_g, C_e, C_h = g_vals.shape[-1], e_vals.shape[-1], hgt_val.shape[-1]
        H, W = self.projector.H, self.projector.W
        
        # Reshape to (B*H*W, T*C)
        g_feat = x_grid[0, :, :C_g].permute(2, 3, 0, 1).reshape(-1, x_grid.shape[1] * C_g)
        e_feat = x_grid[0, :, C_g:C_g+C_e].permute(2, 3, 0, 1).reshape(-1, x_grid.shape[1] * C_e)
        hgt_feat = x_grid[0, :, C_g+C_e:C_g+C_e+C_h].permute(2, 3, 0, 1).reshape(-1, x_grid.shape[1] * C_h)
        
        dense_x = torch.cat([g_feat, e_feat, hgt_feat, dense_geo], dim=-1).cpu().numpy()
        pred_map = self.model.predict(dense_x).T.reshape(6, H, W)
        out_dir = "./results/catboost/inference"
        os.makedirs(out_dir, exist_ok=True)
        plot_on_map(pred_map, resolution=0.005, output_dir=out_dir, file_name=f"{timestamp}(UT)", mute=True)

    def _collect(self, loader):
        X, Y = [], []
        for b in loader:
            x, y = self.batch_process(b)
            if x.size > 0: X.append(x); Y.append(y)
        return np.vstack(X), np.vstack(Y)

    def rolling_validation(self, train_loader, test_loader):
        # 1. Initial training: 2021-2024
        X_all, Y_all = self._collect(train_loader)
        X_all_list = [X_all]
        Y_all_list = [Y_all]
        self.model.fit(X_all, Y_all)
        print("Finish the initial training.")
        
        # 2. Group test data by day (YYYYMMDD)
        dataset = test_loader.dataset
        day_map = defaultdict(list)
        for i, idx in enumerate(dataset.sample_indices):
            ts = dataset.timestamps[idx]
            if isinstance(ts, bytes): ts = ts.decode('utf-8')
            day_map[ts[:8]].append(i)
        
        sorted_days = sorted(day_map.keys())
        all_yt, all_yp = [], []
        monthly_data = defaultdict(lambda: {"yt": [], "yp": []})
        custom_format = "{desc}: {percentage:3.0f}% | {elapsed} | {rate_fmt} | {postfix}"
        pbar = tqdm(total=len(sorted_days), desc="Rolling Validation", leave=True, bar_format=custom_format)
        
        # 3. Expanding Window Loop (Rolling validation)
        for day in sorted_days:
            month_key = day[0:6]
            day_loader = DataLoader(Subset(dataset, day_map[day]), batch_size=1)
            
            X_day_list, Y_day_list = [], []
            for i, batch in enumerate(day_loader):
                x, y = self.batch_process(batch)
                if x.size > 0:
                    X_day_list.append(x); Y_day_list.append(y)
                
                if "20250101" <= day <= "20250110":
                    real_idx = dataset.sample_indices[day_map[day][i]]
                    ts = dataset.timestamps[real_idx]
                    if isinstance(ts, bytes): ts = ts.decode('utf-8')
                    self._dense_inference_and_plot(batch, ts)
                    
            if not X_day_list: continue
            X_day, Y_day = np.vstack(X_day_list), np.vstack(Y_day_list)

            # Predict current day N before training on it
            y_pred = self.model.predict(X_day)
            all_yp.append(y_pred)
            all_yt.append(Y_day)
            monthly_data[month_key]["yp"].append(y_pred)
            monthly_data[month_key]["yt"].append(Y_day)
            
            # Expand training pool and update model
            X_all_list.append(X_day)
            Y_all_list.append(Y_day)
            self.model.fit(np.vstack(X_all_list), np.vstack(Y_all_list))
            pbar.set_postfix_str(f"Date: {day}")
            pbar.update(1)
        pbar.close()

        # 4. Aggregate and Print metrics
        yt, yp = np.vstack(all_yt), np.vstack(all_yp)
        os.makedirs("./results/catboost/", exist_ok=True)
        with open("./results/catboost/CatBoost_metrics.txt", "w", encoding="utf-8") as f:
            def log(msg):
                print(msg)
                f.write(str(msg) + "\n")
            log("Record the results of CatBoost")
            log("------ Regression Model Aggregate Metrics ------")
            for j, target in enumerate(TARGETFEATURES):
                res = self._get_metrics(yt[:, j], yp[:, j])
                log(f"{target} R square: {res[0]:.2f} | MAE: {res[1]:.2f} | RMSE: {res[2]:.2f} | MAPE(%): {res[3]}")
            
            for month in sorted(monthly_data.keys()):
                log(f"\n------ Aggregate Metrics for {month} ------")
                yt_m = np.vstack(monthly_data[month]["yt"])
                yp_m = np.vstack(monthly_data[month]["yp"])
                for j, target in enumerate(TARGETFEATURES):
                    res = self._get_metrics(yt_m[:, j], yp_m[:, j])
                    log(f"{target} R square: {res[0]:.2f} | MAE: {res[1]:.2f} | RMSE: {res[2]:.2f} | MAPE(%): {res[3]}")
            
            for i, fea in enumerate(TARGETFEATURES):
                m, valid_months = 0, []
                MAE, RMSE, R2, MAPE = [], [], [], []
                for month in sorted(monthly_data.keys()):
                    yt_m = np.vstack(monthly_data[month]["yt"])
                    yp_m = np.vstack(monthly_data[month]["yp"])
                    r2, mae, rmse, mape = self._get_metrics(yt_m[:, i], yp_m[:, i])
                    MAE.append(mae); RMSE.append(rmse); R2.append(r2); MAPE.append(mape/100)
                    valid_months.append(m + 1)
                    m += 1
                if valid_months:
                    save_path = f"./results/catboost/{fea}/"
                    plot_monthly_error(fea, MAE, RMSE, R2, MAPE, valid_months, outpath=save_path, model="CatBoost")

        shutil.rmtree("./catboost_info", ignore_errors=True)

    def _get_metrics(self, y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        err = y_true - y_pred
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))
        
        # R2 Score
        ss_res = np.sum(err**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAPE (Handle zero values to avoid division by zero)
        mask = y_true > 1e-6
        mape = np.mean(np.abs(err[mask] / y_true[mask])) * 100 if mask.any() else 0
        
        return np.round(r2, 2), np.round(mae, 2), np.round(rmse, 2), np.round(mape)
        
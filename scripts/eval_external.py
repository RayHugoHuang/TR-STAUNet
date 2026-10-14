import _env
import h5py
import numpy as np
from utils.metrics import Evaluator
from scipy.spatial import KDTree
from configs import cfg

ROOT = "./dataset_storage/processed"
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, RES = 120.0, 122.0, 21.5, 25.5, 0.005
TARGETFEATURES = ["PM2.5", "PM10", "O3", "NO2", "CO", "SO2"]

def coord_to_grid_index(lon, lat) -> tuple[np.array, np.array]:
    return np.round((lon - LON_MIN) / RES).astype(int), np.round((lat - LAT_MIN) / RES).astype(int)

if __name__ == "__main__":
    eva = Evaluator()
    target_folder = "AttentionUnet_20261008"
    model = target_folder.split("_")[0]

    # (IDW + Wind Speed + Terrain) TEPA Method
    with h5py.File(f"{ROOT}/tepa_2025.h5", "r") as hf:
        y_tepa, geo_tepa = hf["data"][:], hf["geo"][:] # (T, 26+75, 6), (26+75, 2)
        tepa_tree = KDTree(geo_tepa)

    # External Reference (LEPB)
    with h5py.File(f"{ROOT}/external_2025.h5", "r") as hf:
        y_ex, y_geo = hf["data"][:], hf["geo"][:] # (T, 26, 6), (26, 2)
        all_ts = [t.decode("utf-8") for t in hf["timestamps"]]
        x_idx, y_idx = coord_to_grid_index(y_geo[:, 0], y_geo[:, 1])
        external_idx = tepa_tree.query(y_geo)[1]

    # Model Inference
    with h5py.File(f"./results/{target_folder}/{model}.h5", "r") as hf:
        valid_ts = [t.decode("utf-8") for t in hf["timestamps"]]
        # Load only timestep t=N to avoid allocating full 5D tensor
        t_indices = np.searchsorted(all_ts, valid_ts)
        y_pred_eval = np.empty((len(valid_ts), 6, len(y_geo)), dtype=np.float32)
        for i in range(0, len(valid_ts), chunk_size := 100):
            y_pred_eval[i:i + chunk_size] = hf["y_pred"][i:i + chunk_size, 0][..., y_idx, x_idx]
        y_pred_eval = y_pred_eval[..., np.newaxis] # (B, T, C, H, W) -> (B, C, H, W)

    y_ex_eval = y_ex[t_indices].transpose(0, 2, 1)[..., np.newaxis]
    y_mask_eval = ~np.isnan(y_ex_eval)
    y_tepa_eval = y_tepa[t_indices][:, external_idx].transpose(0, 2, 1)[..., np.newaxis]
    mask = y_mask_eval & ~np.isnan(y_tepa_eval)

    with open(f"./{target_folder}.txt", "w", encoding="utf-8") as f:
        f.write(f"-------------------- {model} vs local stations --------------------\n")
        eva.all_errors(TARGETFEATURES, y_ex_eval, y_pred_eval, y_mask_eval, log_file=f)
        f.write("\n-------------------- TEPA vs local stations --------------------\n")
        eva.all_errors(TARGETFEATURES, y_ex_eval, y_tepa_eval, mask, log_file=f)

    # Ground Truth (TAQMN)
    with h5py.File(cfg.GT_PATH, "r") as hf:
        y_gt, geo_gt = hf["data"][:], hf["geo"][:] # (T, 75, 6), (75, 2)
        ts = [t.decode("utf-8") for t in hf["timestamps"]]
        gt_indices_ts = np.searchsorted(ts, valid_ts)
        gt_idx = tepa_tree.query(geo_gt)[1]

    y_true_eval = y_gt[gt_indices_ts].transpose(0, 2, 1)[..., np.newaxis]
    y_mask_eval = ~np.isnan(y_true_eval)
    y_tepa_eval = y_tepa[t_indices][:, gt_idx].transpose(0, 2, 1)[..., np.newaxis]
    mask = y_mask_eval & ~np.isnan(y_tepa_eval)

    with open("./tepa_log.txt", "w", encoding="utf-8") as f:
        f.write("-------------------- TEPA vs gt stations --------------------\n")
        eva.all_errors(TARGETFEATURES, y_true_eval, y_tepa_eval, mask, log_file=f)
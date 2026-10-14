import torch, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
# ========================= Belows are custom modules in this project =========================
from .engine import batch_process
from utils import MaskedL1Loss
from configs import cfg

# --- Parameters ---
CHECK_SAMPLES = 200
OUTPUT_DIR = "./results/"

def plot_global_results(avg_diffs, input_names, top_n=cfg.INPUT_CHANNELS, name="feature_importance", out_dir=OUTPUT_DIR):
    """ Plots the global feature importance bar chart. """
    indices = np.argsort(avg_diffs)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance (Global Ablation) - Top {top_n}")
    plt.barh(range(top_n), avg_diffs[indices[:top_n]], align='center')
    plt.yticks(range(top_n), [input_names[i] for i in indices[:top_n]])
    plt.xlabel("Increase in Loss (MAE)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.xlim(0, int(avg_diffs[indices[0]])+1)
    plt.savefig(f"{out_dir}/{name}_global.png", dpi=600, bbox_inches='tight')
    plt.close()

def plot_detailed_results(importance_matrix, input_names, target_names, top_n=15, name="feature_importance", out_dir=OUTPUT_DIR):
    """ Plots feature importance per target channel. """
    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for t_idx, target in enumerate(target_names):
        ax = axes[t_idx]
        scores = importance_matrix[:, t_idx]
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_scores = scores[top_indices]
        top_names = [input_names[i] for i in top_indices]
        
        ax.barh(range(top_n), top_scores, color='skyblue', align='center')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_title(f"Features for {target}")
        if t_idx >= 3:
            ax.set_xlabel("Increase in Loss (MAE)")

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.93, wspace=0.3, hspace=0.12)
    plt.savefig(f"{out_dir}/{name}_detailed.png", dpi=600, bbox_inches='tight')
    plt.close()

def plot_temporal_results(avg_diffs, seq_len, name="temporal_importance", out_dir=OUTPUT_DIR):
    """ Plots the temporal feature importance bar chart. """
    plt.figure(figsize=(10, 6))
    plt.title("Temporal Importance (Global Ablation)")
    plt.bar(range(seq_len), avg_diffs, align='center', color='coral')
    plt.xticks(range(seq_len), [f"Step {i}" for i in range(seq_len)])
    plt.ylabel("Increase in Loss (MAE)")
    plt.xlabel("Time Step (0 is Earliest)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}_temporal.png", dpi=600, bbox_inches='tight')
    plt.close()

@torch.inference_mode()
def _run_ablation_core(model, loader, projector, dim, num_steps, mode='global'):
    """ Universal evaluation core for both feature (dim=2) and temporal (dim=1) ablation. """
    dataset = loader.dataset
    random.seed(42)
    rand_idx = random.sample(range(len(dataset)), min(CHECK_SAMPLES, len(dataset)))
    rand_loader = DataLoader(Subset(dataset, rand_idx), batch_size=1)

    # Initialize Accumulators and loss function
    criterion = MaskedL1Loss("mean" if mode == 'global' else "none")
    shape = num_steps if mode == 'global' else (num_steps, cfg.OUTPUT_CHANNELS)
    diffs = torch.zeros(shape, device=cfg.DEVICE)
    counter = 0 if mode == 'global' else torch.zeros(cfg.OUTPUT_CHANNELS, device=cfg.DEVICE)

    for batch in tqdm(rand_loader, total=len(rand_loader)):
        pb = batch_process(batch, projector, cfg.to_device)
        # Baseline Prediction, flatten: (B, T, C, H, W) -> (B*T, C, H, W)
        x_grid, y_g, y_m = pb.x_grid, pb.y_grid.flatten(0, 1), pb.y_mask.flatten(0, 1)
        base_metric = criterion(model(x_grid).flatten(0, 1), y_g, y_m)
        
        if mode == 'global':
            if torch.isnan(base_metric): continue
            counter += 1
        else:
            valid_channels = (y_m.sum(dim=(0, 2, 3)) > 0) & ~torch.isnan(base_metric)
            counter += valid_channels.float()

        for idx in range(num_steps):
            orig = x_grid.select(dim, idx).clone()
            x_grid.select(dim, idx).zero_()
            abl_metric = criterion(model(x_grid).flatten(0, 1), y_g, y_m)
            x_grid.select(dim, idx).copy_(orig)
            
            diff = torch.nan_to_num(abl_metric - base_metric, nan=0.0)
            if mode != 'global': diff *= valid_channels
            diffs[idx] += diff

    return ((diffs / counter) if mode == 'global' else (diffs / counter.clamp(min=1).unsqueeze(0))).cpu().numpy()

def run_ablation_study(model, loader, projector, mode='global', figname="feature_importance", out_dir=OUTPUT_DIR):
    avg_diffs = _run_ablation_core(model, loader, projector, dim=2, num_steps=cfg.INPUT_CHANNELS, mode=mode)
    if mode == 'global':
        plot_global_results(avg_diffs, cfg.FEATURES, name=figname, out_dir=out_dir)
    else:
        plot_detailed_results(avg_diffs, cfg.FEATURES, cfg.TARGETFEATURES, name=figname, out_dir=out_dir)

def run_temporal_ablation_study(model, loader, projector, figname, out_dir):
    avg_diffs = _run_ablation_core(model, loader, projector, dim=1, num_steps=cfg.SEQ_LEN, mode='global')
    plot_temporal_results(avg_diffs, cfg.SEQ_LEN, name=figname, out_dir=out_dir)

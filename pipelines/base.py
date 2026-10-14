import os, torch, gc, h5py, time
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import autocast
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import (CostSensitiveL1Loss, Augmentation, SpatialProjector, transmission_resistance_loss, 
                    temporal_consistency_loss, plot_loss, plot_on_map, plot_distribution)
from .engine import batch_process, evaluate_and_log
from .ablation import run_ablation_study, run_temporal_ablation_study

class BaseTrainer:
    def __init__(self, cfg, projector, model, model_name, model_path, loaders, mask_mode, use_amp=True, out_dir="./results/"):
        self.cfg = cfg
        self.projector = projector
        self.device = cfg.DEVICE
        self.to_device = cfg.to_device
        self.projector.interpolation = not mask_mode
        self.y_projector = SpatialProjector(res=0.1, static_path="").to(self.device)
        
        self.train_loader, self.val_loader, self.test_loader = loaders if loaders else (None, None, None)
        self.model = model
        self.model_name = model_name
        self.model_path = model_path
        
        self.criterion = CostSensitiveL1Loss()
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4, fused=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, eta_min=1e-5, T_max=cfg.EPOCHS)
        self.use_amp = use_amp
        self.aug = Augmentation(cfg.FILL_VALUE)
        self.best_loss = float('inf')
        self.inference_limit_date = pd.to_datetime(cfg.INFERENCE_LIMIT_DATE)
        self.do_plot = False
        self.gt_dir = out_dir
        self.out_dir = os.path.join(out_dir, f"{self.model_name}_{pd.Timestamp.now().strftime('%Y%m%d')}")
        self.fig_dir = os.path.join(self.out_dir, "inference")
        self.log_dir = os.path.join(self.out_dir, "logs")
        self.training_log = os.path.join(self.log_dir, "training_log.txt")
        for d in (self.fig_dir, self.log_dir): os.makedirs(d, exist_ok=True)
        cfg.save_experiment_config(os.path.join(self.out_dir, "config.json"))

    def train(self, do_analysis=False):
        loss_record, counter = [], 0
        with open(self.training_log, "w", encoding="utf-8") as f:
            f.write(f"Model: {self.model_name}  |  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        for epoch in range(self.cfg.EPOCHS):
            train_loss, val_loss = self._train_epoch(epoch)
            self.scheduler.step()
            loss_record.append([train_loss, val_loss])
            
            if val_loss < self.best_loss:
                counter = 0
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
            else:
                counter += 1
                if counter >= self.cfg.EARLY_STOPPING_TOLERANCE: break

        if loss_record: plot_loss(loss_record, "MAE", self.model_name, self.log_dir)
        print(f"Testing set Loss: {self._val_epoch('test'):.3f}")
        if do_analysis: self._run_ablation()
        torch.cuda.empty_cache()
        gc.collect()
    
    def _log_and_visualize_eval(self, y_pred_all, y_true_all, y_mask_all, timestep, suffix, header_msg, t_step=1):
        log_file = f"{self.log_dir}/{self.model_name}{suffix}.txt"
        out_path = f"{self.out_dir}/step_{t_step}/"
        evaluate_and_log(self.cfg.TARGETFEATURES, y_true_all, y_pred_all, y_mask_all, 
                         timestep, self.model_name, log_file, out_path, header_msg)

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        test_dataset = self.test_loader.dataset
        ts_idx = test_dataset.sample_indices
        print(f"Start ts_idx: {min(ts_idx)}, end ts_idx: {max(ts_idx)}")
        all_timestep = test_dataset.timestamps
        timestep = [all_timestep[ts] for ts in ts_idx]
        idx = timestep.index(self.cfg.TARGET_TIME)
        n, h, w = len(test_dataset.sample_indices), self.projector.H, self.projector.W
        t_len = self.cfg.T_LEN
        
        target_shape, chunk_shape = (n, t_len, 6, h, w), (1, t_len, 6, h, w)
        hf_eval = h5py.File(os.path.join(self.out_dir, "eval_cache.h5"), "w")
        y_pred_all = hf_eval.create_dataset("y_pred", shape=target_shape, chunks=chunk_shape, dtype=np.float32, compression="lzf")
        y_true_all = hf_eval.create_dataset("y_true", shape=target_shape, chunks=chunk_shape, dtype=np.float32, compression="lzf")
        y_mask_all = hf_eval.create_dataset("y_mask", shape=target_shape, chunks=chunk_shape, dtype=bool, compression="lzf")
        buf_mgr = HDF5BufferManager((y_pred_all, y_true_all, y_mask_all), n, t_len, h, w)
        ts_to_saved = []

        for i, batch in enumerate(tqdm(self.test_loader, desc=f"Inference ({self.model_name})", leave=True)):
            y_pred, y_true, y_mask = self._predict_step(batch)
            self._plot_prediction(t_len, timestep[i], y_pred, ts_to_saved, "")
            buf_mgr.write(y_pred, y_true, y_mask, i)

            if i == idx:
                pb = batch_process(batch, self.y_projector, self.to_device)
                y_target = pb.y_grid.squeeze(0).cpu().numpy()
                y_target[y_target == 0] = np.nan
        
        plot_on_map(y_target[0], resolution=0.1, output_dir=self.gt_dir, file_name=f"GT{timestep[idx]}(10)(UT)", mute=False)
        for t in range(t_len):
            header_msg = f"Record the results of {self.model_name} (Step {t+1})\n------ Aggregate Metrics ------"
            self._log_and_visualize_eval(y_pred_all[:, t], y_true_all[:, t], y_mask_all[:, t], timestep, 
                                            f"_step{t+1}", header_msg, t+1)
        
        hf_eval.close()
        if 'y_target' in locals(): del y_target
        os.remove(os.path.join(self.out_dir, "eval_cache.h5"))
        torch.cuda.empty_cache()

    def evaluate_adaptive(self, online_set, rolling_days=3, initial_date="20250101"):
        self.model.eval()
        day_map = defaultdict(list)
        for i, idx in enumerate(online_set.sample_indices):
            day_map[online_set.timestamps[idx][:8]].append(i)
        
        unique_days = sorted(list(day_map.keys()))
        init_date = pd.to_datetime(initial_date)
        eval_indices = [idx for d in unique_days if pd.to_datetime(d) >= init_date for idx in day_map[d]]
        all_timestep = online_set.timestamps
        timestep = [all_timestep[online_set.sample_indices[i]] for i in eval_indices]
        idx = timestep.index(self.cfg.TARGET_TIME)
        n, h, w = len(eval_indices), self.projector.H, self.projector.W
        t_len = self.cfg.T_LEN
        
        temp_path = os.path.join(self.out_dir, f"{init_date.year}_temp.h5")
        target_shape, chunk_shape = (n, t_len, 6, h, w), (1, t_len, 6, h, w)
        hf_cache = h5py.File(os.path.join(self.out_dir, f"{self.model_name}.h5"), "w")
        hf_temp = h5py.File(temp_path, "w")
        y_pred_all = hf_cache.create_dataset("y_pred", shape=target_shape, chunks=chunk_shape, dtype=np.float32, compression="lzf")
        y_true_all = hf_temp.create_dataset("y_true", shape=target_shape, chunks=chunk_shape, dtype=np.float32, compression="lzf")
        y_mask_all = hf_temp.create_dataset("y_mask", shape=target_shape, chunks=chunk_shape, dtype=bool, compression="lzf")
        buf_mgr = HDF5BufferManager((y_pred_all, y_true_all, y_mask_all), n, t_len, h, w)
        ts_to_saved = []
        
        init_weights = torch.load(self.model_path, map_location=self.device)
        pbar = tqdm(total=n, desc=f"Rolling Eval ({self.model_name})", leave=True)
        ptr = 0

        finetune_optimizer = AdamW(self.model.parameters(), lr=self.cfg.FINETUNE_LR, fused=True, weight_decay=1e-4)
        for date_str in unique_days:
            target_date = pd.to_datetime(date_str)
            if target_date < init_date: continue
            win_days = pd.date_range(target_date - pd.Timedelta(days=rolling_days), periods=rolling_days).strftime("%Y%m%d")
            mini_idx = [idx for day in win_days if day in day_map for idx in day_map[day]]
            target_date_idx = day_map[date_str]
            
            if mini_idx:
                loader = DataLoader(Subset(online_set, mini_idx), 1, shuffle=True)
                self._finetune(loader, finetune_optimizer)

            pred_loader = DataLoader(Subset(online_set, target_date_idx), 1)
            for batch in pred_loader:
                y_pred, y_true, y_mask = self._predict_step(batch)
                self._plot_prediction(t_len, timestep[ptr], y_pred, ts_to_saved)
                buf_mgr.write(y_pred, y_true, y_mask, ptr)
                pbar.set_postfix_str(f"Date: {date_str}")
                pbar.update(1)
                ptr+=1
        pbar.close()
        self.model.load_state_dict(init_weights)
        
        hf_cache.create_dataset("timestamps", data=np.array(ts_to_saved, dtype='S'), compression="lzf")
        for t in range(t_len):
            header_msg = f"------ Online Fine-tuning Aggregate Metrics (Step {t+1}) ------"
            #plot_distribution(y_pred_all[:, t], y_true_all[:, t], y_mask_all[:, t], self.cfg.TARGETFEATURES)
            self._log_and_visualize_eval(y_pred_all[:, t], y_true_all[:, t], y_mask_all[:, t], timestep, 
                                         f"_rolling_{rolling_days}_step{t+1}", header_msg, t+1)
        
        hf_temp.close()
        os.remove(temp_path)
        hf_cache.close()
        torch.cuda.empty_cache()

    def _finetune(self, train_loader, optimizer):
        for batch in train_loader:
            pb = batch_process(batch, self.projector, self.to_device, project_uv=True)
            self._train_step(pb, optimizer, clip_norm=2.0)

    def _plot_prediction(self, t_len, timestep, y_pred, ts_to_saved, dir_suffix="_online_finetune"):
        base_time = pd.to_datetime(timestep, format="%Y%m%d%H%M") # UT
        for t in range(t_len):
            fig_path = os.path.join(self.fig_dir, f"{self.model_name}{dir_suffix}", f"step_{t+1}")
            os.makedirs(fig_path, exist_ok=True)
            target_time = base_time + pd.Timedelta(hours=t+1)
            if target_time < self.inference_limit_date and self.do_plot:
                plot_on_map(y_pred[t], output_dir=fig_path, file_name=target_time.strftime("%Y%m%d%H%M(UT)"), mute=True)
        
        ts_to_saved.append((base_time + pd.Timedelta(hours=8+1)).strftime("%Y%m%d%H%M")) # save as LT

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, count = 0, 0
        shift, drop = 40, 0.5
        custom_format = "{desc}: {percentage:3.0f}% | {elapsed}<{remaining} | {rate_fmt} | {postfix}"
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.EPOCHS}", bar_format=custom_format)

        for batch in pbar:
            pb = batch_process(batch, self.projector, self.to_device, self.aug, shift, drop, project_uv=True)
            loss_val = self._train_step(pb, self.optimizer, clip_norm=max(2.0, 10.0 * (0.9 ** epoch)))
            if loss_val is None: continue
            count += 1
            total_loss += loss_val
            pbar.set_postfix_str(f"Loss: {total_loss / count:.3f}")

            if count == len(self.train_loader):
                avg_val_loss = self._val_epoch()
                pbar.set_postfix_str(f"Loss: {total_loss / count:.3f} | val Loss: {avg_val_loss:.3f}")
        
        with open(self.training_log, "a", encoding="utf-8") as f:
            f.write(f"{str(pbar)} | time: {time.strftime('%Y-%m-%d %H:%M:%S')} \n")
        return total_loss / count, avg_val_loss
    
    def _train_step(self, pb, optimizer, clip_norm):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
            pred = self.model(pb.x_grid)
        pred = pred.float()
        tc_loss = temporal_consistency_loss(pred, pb.y_grid, pb.y_mask)

        pred = pred.flatten(0, 1) # (B, T, C, H, W) -> (B*T, C, H, W)
        y, y_m = pb.y_grid.flatten(0, 1), pb.y_mask.flatten(0, 1)
        uv, uv_m = pb.uv_grid.flatten(0, 1), pb.uv_mask.flatten(0, 1)

        loss = self.criterion(pred, y, y_m)
        if torch.isnan(loss): return None
        base_loss = loss.item()
        mixed = (pred * (1 - y_m)) + (y * y_m)
        loss = loss + transmission_resistance_loss(mixed, y_m, uv, uv_m) + tc_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
        optimizer.step()
        return base_loss
    
    @torch.inference_mode()
    def _val_epoch(self, mode="val"):
        self.model.eval()
        total_loss = 0.0
        loader = self.val_loader if mode=="val" else tqdm(self.test_loader, desc="Inference on testing set")
        for batch in loader:
            pb = batch_process(batch, self.projector, self.to_device)
            pred = self.model(pb.x_grid)
            pred = pred.flatten(0, 1) # (B, T, C, H, W) -> (B*T, C, H, W)
            y, y_m = pb.y_grid.flatten(0, 1), pb.y_mask.flatten(0, 1)
            loss = self.criterion(pred, y, y_m)
            total_loss += loss.item()
        return total_loss / len(loader)
    
    @torch.inference_mode()
    def _predict_step(self, batch):
        pb = batch_process(batch, self.projector, self.to_device)
        pred = self.model(pb.x_grid)
        return pred.squeeze(0).cpu().numpy(), pb.y_grid.squeeze(0).cpu().numpy(), pb.y_mask.squeeze(0).cpu().numpy().astype(bool)
    
    @torch.inference_mode()
    def _run_ablation(self):
        run_temporal_ablation_study(self.model, self.test_loader, self.projector, self.model_name, self.out_dir)
        run_ablation_study(self.model, self.test_loader, self.projector, 'global', self.model_name, self.out_dir)
        run_ablation_study(self.model, self.test_loader, self.projector, 'detailed', self.model_name, self.out_dir)

class HDF5BufferManager:
    def __init__(self, datasets, n, t_len, h, w, buffer_size=306):
        self.y_pred_all, self.y_true_all, self.y_mask_all = datasets
        self.buffer_size, self.n = min(buffer_size, n), n
        
        # Initialize pre-allocated buffer to reduce Disk I/O and memory overhead
        self.p_buf = np.empty((self.buffer_size, t_len, 6, h, w), dtype=np.float32)
        self.t_buf = np.empty((self.buffer_size, t_len, 6, h, w), dtype=np.float32)
        self.m_buf = np.empty((self.buffer_size, t_len, 6, h, w), dtype=bool)
        self.buf_start, self.buf_idx = 0, 0

    def write(self, y_pred, y_true, y_mask, current_i):
        # Write to pre-allocated buffer
        self.p_buf[self.buf_idx] = y_pred
        self.t_buf[self.buf_idx] = y_true
        self.m_buf[self.buf_idx] = y_mask
        self.buf_idx += 1

        # Write to HDF5 when buffer is full or at the last element
        if self.buf_idx == self.buffer_size or current_i == self.n - 1: 
            buf_end = self.buf_start + self.buf_idx
            self.y_pred_all[self.buf_start:buf_end] = self.p_buf[:self.buf_idx]
            self.y_true_all[self.buf_start:buf_end] = self.t_buf[:self.buf_idx]
            self.y_mask_all[self.buf_start:buf_end] = self.m_buf[:self.buf_idx]
            self.buf_start = buf_end
            self.buf_idx = 0
import torch, gc
import pandas as pd
from torch.utils.data import DataLoader, random_split
from pipelines import BaseTrainer, AttenUNetTrainer, Trainer_CatBoost, data_preprocessing_pipeline
from data import SparseDataGenerator, sparse_collate
from utils import SpatialProjector
from configs import cfg

def run_experiment(trainer: BaseTrainer, online_set: SparseDataGenerator) -> None:
    trainer.do_plot = True
    trainer.train(do_analysis=True)           # Training model
    trainer.evaluate()                        # Evaluation
    trainer.evaluate_adaptive(online_set, cfg.ROLLING_DAYS, cfg.TEST_START_DATE) # Evaluation with rolling fine-tune

if __name__ == "__main__":
    data_preprocessing_pipeline() # It won't overwrite the existing files.
    # Accelerating
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True

    # Dataset and Loader
    projector = SpatialProjector().to(cfg.DEVICE)
    common_paths = [cfg.GEMS_PATH, cfg.ERA5_PATH, cfg.GT_PATH]
    full_set = SparseDataGenerator(*common_paths, cfg.START_DATE, cfg.TRAIN_END_DATE, cfg.STD_PATH)
    tra_size, val_size = int(len(full_set) * cfg.TRAIN_VAL_SPLIT), len(full_set) - int(len(full_set) * cfg.TRAIN_VAL_SPLIT)
    train_set, valid_set = random_split(full_set, [tra_size, val_size], generator=torch.Generator().manual_seed(cfg.SEED))
    print(f"Total: {len(full_set)} | train : val = {len(train_set)} : {len(valid_set)}")

    kwargs = {"collate_fn":sparse_collate, "num_workers":4, "pin_memory":True, "prefetch_factor":2, "persistent_workers":True}
    train_loader = DataLoader(train_set, cfg.BATCH_SIZE, True, **kwargs)
    valid_loader = DataLoader(valid_set, cfg.BATCH_SIZE, False, **kwargs)
    # Testing set
    test_set = SparseDataGenerator(*common_paths, cfg.TEST_START_DATE, cfg.END_DATE, cfg.STD_PATH)
    test_loader = DataLoader(test_set, 1, False, num_workers=4, prefetch_factor=2, pin_memory=True)
    online_start_date = (pd.to_datetime(cfg.TEST_START_DATE) - pd.Timedelta(days=cfg.ROLLING_DAYS)).strftime("%Y%m%d")
    online_set = SparseDataGenerator(*common_paths, online_start_date, cfg.END_DATE, cfg.STD_PATH)
    all_loaders = (train_loader, valid_loader, test_loader)

    # Main study model and validation
    study = AttenUNetTrainer(cfg, projector, all_loaders, use_amp=True, late_fusion=False)
    run_experiment(study, online_set)
    del study
    gc.collect()
    torch.cuda.empty_cache()
    
    # Baseline model and validation
    catboost = Trainer_CatBoost()
    #catboost.rolling_validation(train_loader, test_loader)
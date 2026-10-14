from data.preprocessing import SparseTargetBuilder, get_gems, get_era5
from configs import cfg
import os

# The pipeline won't overwrite the existing files.
def data_preprocessing_pipeline():
    from data import compute_and_save_stats, build_static_features
    # Create *GEMS_sparse*.h5
    if not os.path.exists(cfg.GEMS_PATH):
        get_gems(cfg.START_DATE, cfg.END_DATE, cfg.GEMS_PATH)
    else:
        print(f"'{os.path.basename(cfg.GEMS_PATH)}' already exists.")
    # Create *ERA5_sparse*.h5
    if not os.path.exists(cfg.ERA5_PATH):
        get_era5(cfg.START_DATE, cfg.END_DATE, cfg.ERA5_PATH)
    else:
        print(f"'{os.path.basename(cfg.ERA5_PATH)}' already exists.")
    # Create *sparse_targets*.h5
    if not os.path.exists(cfg.GT_PATH):
        gt_builder = SparseTargetBuilder()
        gt_builder.build(site_csv="./dataset_storage/site_coordinate.csv",
                gt_root_dir="./dataset_storage/raw/gt_data/", 
                output_h5=cfg.GT_PATH, start_date=cfg.START_DATE+"00", end_date=cfg.END_DATE+"23")
    else:
        print(f"'{os.path.basename(cfg.GT_PATH)}' already exists.")
    # Create *MeanStd*.h5
    compute_and_save_stats(cfg.GEMS_PATH, cfg.ERA5_PATH, cfg.STD_PATH, cfg.GT_PATH, cfg.START_DATE, cfg.TRAIN_END_DATE)
    # Create static_features.h5
    build_static_features()
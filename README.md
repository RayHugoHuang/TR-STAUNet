# Physics-Informed Deep Spatiotemporal Downscaling of GEMS Geostationary Satellite Observations for Short-Term Multi-Pollutant Forecasting: Official PyTorch Implementation

This repository provides the official PyTorch implementation for physics-informed spatiotemporal downscaling and air quality forecasting across Taiwan (0.005° / ~500m grid). Integrating GEMS satellite observations, ERA5-Land reanalysis, and physics-guided atmospheric transport constraints, the framework delivers hourly predictions for six criteria air pollutants (PM2.5, PM10, O3, NO2, CO, SO2).

## 📂 Directory Structure

```text
TR-STAUNet/
├── configs/                  # Data and related parameter settings
│   ├── __init__.py
│   ├── config.py             # Read and modulize the information from config.yaml
│   └── config.yaml           # All settings in this project
│
├── data/                     # [Data pipeline] Responsible for data reading and preprocessing
│   ├── __init__.py
│   ├── dataset.py            # Define the behavior of Dataset and DataLoader
│   ├── standardization.py    # Data Standardization
│   ├── static_features.py    # Static feature definition
│   └── preprocessing/        # [Submodule] Independent web crawlers and file conversion scripts
│       ├── __init__.py       # Make "preprocessing" into modules
│       ├── get_era5.py       # Preprocessing ERA5 data
│       ├── get_gems.py       # Preprocessing GEMS data
│       └── get_gt.py         # Preprocessing Ground Truth data
│
├── dataset_storage/          # Data storage area
│   ├── raw/                  # Original download data
│   ├── processed/            # Processed .h5 files for training (Sample dataset, full multi-year Git Ignored)
│   └── site_coordinate.csv   # Geographic coordinate data from ground observation stations
│
├── models/                   # Artificial neural network architecture definition
│   ├── __init__.py
│   ├── attenunet.py          # U-Net architecture with Attention Mechanism
│   └── layers.py             # Basic convolutional layers and related objects
│
├── pipelines/                # [Execution Logic] Encapsulated training and prediction pipeline
│   ├── __init__.py
│   ├── base.py               # Base training object and methods, including training, validation, finetune and prediction
│   ├── benchmark.py          # CatBoost model benchmark
│   ├── engine.py             # functions for iteration and fine-tuning
│   ├── ablation.py           # Analyzing feature and temporal importance of the model
│   ├── preprocess.py         # Define data_preprocessing_pipeline()
│   └── trainer.py            # Model-specific trainers inherited from base.py
│
├── scripts/                  # Standalone diagnostic, profiling, and evaluation tools
│   ├── _env.py               # Shared environment setup and root path injection
│   ├── data_profiler.py      # Feature completeness and comparative statistics profiling
│   ├── eval_external.py      # Metric evaluation on external ground stations
│   └── visualize_features.py # Dynamic projection and feature visualization tool
│
├── results/                  # Model validation results (Git Ignored)
│
├── utils/                    # [General Tools] Model-independent helper functions
│   ├── __init__.py
│   ├── common.py             # Augmentation, load pre-trained weights
│   ├── loss.py               # Loss function (Include regularizer)
│   ├── metrics.py            # Evaluation metrics (Evaluator)
│   ├── projector.py          # Space projection module
│   └── visualization.py      # Plotting and visualization tools
│
├── weights/                  # Store training weights .pth (Git Ignored)
├── bootstrap.py              # Environment setting and folder creation
├── main.py                   # Main execution sequence
├── README.md                 # Project Description Document
└── requirements.txt          # Necessary modules and dependencies
```
## 🛠️ Environment & Quickstart

### Prerequisites
*Note: If your hardware requires a different version, please visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to find the correct installation command.*
* **OS**: Windows (Tested on Windows 10)
* **Python**: 3.9+
* **CUDA / GPU**: 12.9 (16GB VRAM for default `batch_size=4`; ≥4GB requires setting `batch_size=1` in `configs/config.yaml`; CPU fallback supported)
* **PyTorch**: 2.8.0+cu129 (Tested on NVIDIA RTX 2000 Ada)

### Quickstart (Installation & Verification)

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd TR-STAUNet
   ```
2. **Create & activate environment**
   ```bash
   conda create -n air_pred python=3.9 -y
   conda activate air_pred
   ```
3. **Initialize directories and install dependencies**
   ```bash
   python bootstrap.py
   ```
   *(If automated PyTorch installation fails: `pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129`)*

4. **Run pipeline on bundled sample dataset**
   ```bash
   # Requires ~16GB VRAM. If encountering CUDA OOM, reduce batch_size in configs/config.yaml
   python main.py
   ```
   *(Executes automated data preprocessing, deep neural network training, evaluation, ablation analysis, and adaptive online fine-tuning on the pre-packaged 1-month sample dataset, the results will generate in `results/AttentionUnet_yyyymmdd`).*

## 📦 Dataset & Specifications

### 1. Data Directory Hierarchy
Place or link raw data files according to the hierarchy below:

```text
dataset_storage/
├── site_coordinate.csv                     # Station coordinates [SiteId, lon, lat]
│
├── raw/                                    # Raw source files (Static maps bundled; others downloaded)
│   ├── gt_data/                            # Ground station hourly observations
│   │   ├── yyyy/                           # 4-digit year folders with *.csv per station (e.g., 2021/<station_name>.csv)
│   │   └── ...
│   ├── ERA5/                               # ERA5-Land climate NetCDF files (*.nc)
│   ├── TEDS13.h5                           # Emission inventory static features (Bundled, ~1.6 MB)
│   ├── landuse_pop.h5                      # Land use & population density features (Bundled, ~2.2 MB)
│   └── dsm_500m.h5                         # Digital Surface Model elevation features (Bundled, ~0.5 MB)
│
└── processed/                              # Processed tensors (Sample dataset bundled; full datasets Git Ignored)
    ├── <yy>_MeanStd_<yy>.h5                # Normalization stats (*MeanStd*.h5, 2-digit years, e.g., 25_MeanStd_25.h5 or 21_MeanStd_24.h5)
    ├── <yymmdd>_GEMS_sparse_<yymmdd>.h5    # Projected GEMS features (*_GEMS_sparse_*.h5, 6-digit dates)
    ├── <yymmdd>_ERA5_sparse_<yymmdd>.h5    # Projected ERA5 features (*_ERA5_sparse_*.h5, 6-digit dates)
    ├── sparse_targets_<yyyy>_<yyyy>.h5     # Aligned ground truth targets (*sparse_targets*.h5, 4-digit years)
    ├── static_features.h5                  # Combined topography & emission static features (Auto-built by pipeline)
    ├── tepa_<yyyy>.h5                      # Point-extracted TEPA baseline (101 station coordinates: 26 LEPB + 75 TAQMN)
    └── external_<yyyy>.h5                  # (Optional) External reference station observations (26 municipal LEPB stations)

<gems_dir> / <gems_2025_dir>/               # Local or external storage for GEMS satellite scans
└── L2_<Product>/                           # e.g., L2_AERAOD, L2_O3T, L2_NO2
    └── yyyymm/                             # 4-digit year + 2-digit month
        └── dd/                             # 2-digit day
            └── *ORI.nc
```

*(Note: Processed filenames containing date placeholders (`<yy>`, `<yyyy>`, `<yymmdd>`) are dynamically configured under `paths: processed_files` in `configs/config.yaml` to match your designated training/testing intervals.)*

### 2. Sample Dataset (Quickstart) vs. Full Dataset

This repository supports two execution modes:

#### A. Sample Dataset (Pre-Packaged Quickstart)
For quickstart execution, smoke testing, and pipeline verification without downloading raw satellite and reanalysis data, the repository comes bundled with a lightweight 1-month spatiotemporal slice (**January 2025**, ~75 MB total) under `dataset_storage/`:
* **Raw Static Maps** (`dataset_storage/raw/`, ~4.4 MB total):
  * `TEDS13.h5` (~1.6 MB): TEDS 13.0 gridded emissions inventory.
  * `landuse_pop.h5` (~2.2 MB): 9 land-use categories & population density.
  * `dsm_500m.h5` (~0.5 MB): 500m Digital Surface Model elevation.
  *(Note: `dataset_storage/processed/static_features.h5` is automatically compiled from these three raw maps in <1 second upon running the pipeline).*
* **Processed Spatiotemporal Tensors** (`dataset_storage/processed/`, ~70 MB total):
  * `250101_GEMS_sparse_250131.h5` (~33.6 MB): Hourly GEMS satellite observations for Jan 2025.
  * `250101_ERA5_sparse_250131.h5` (~6.7 MB): Hourly ERA5-Land surface reanalysis for Jan 2025.
  * `sparse_targets_2025_2025.h5` (~1.3 MB): Ground truth target observations for 75 national stations (Jan 2025).
  * `25_MeanStd_25.h5` (~10 KB): Channel-wise mean and standard deviation normalization statistics (Jan 2025).
  * `tepa_2025.h5` (~20.8 MB): Official TEPA baseline values extracted at 101 station coordinates (Full Year 2025).
  * `external_2025.h5` (~5.8 MB): Ground observations across 26 independent municipal LEPB stations (Full Year 2025).

*(Note: `configs/config.yaml` is pre-configured out-of-the-box for this sample dataset. See [Usage & Advanced Experiments](#-usage--advanced-experiments) below to run training, evaluation, and diagnostic tools).*

#### B. Full Multi-Year Dataset (2021–2025 Benchmark Reproduction)
To reproduce the full multi-year benchmarks and experimental results reported in the paper (or to run inference using pre-trained weights):
1. **Download Full Processed Archive & Pre-Trained Weights:**
   * Download the complete multi-year processed dataset archive (~2.7 GB) and pre-trained checkpoint (`weights/AttentionUnet.pth`) from the [Google Drive Repository](https://drive.google.com/drive/folders/1MLcSYo2AmH2kK__gj1W3rXUQz1ZVjrx0?usp=drive_link).
   * Place the processed `.h5` files into `dataset_storage/processed/` and the checkpoint into `weights/`.
   * *(Alternatively, retrieve the raw GEMS/ERA5/ground truth datasets as described in Sections 3–5 below and execute the end-to-end preprocessing pipeline).*
2. **Configure `configs/config.yaml` for Full Multi-Year Runs:**
   Update the date split and processed file parameters in `configs/config.yaml`:
   ```yaml
   # Multi-year training & testing date split
   start_date: "2021-01-01"
   end_date: "2025-12-31"
   train_end_date: "2024-12-31"
   test_start_date: "2025-01-01"
   inference_limit_date: "2025-01-11 00:00"

   # Processed tensor filenames
   paths:
     processed_files:
       gems: "210101_GEMS_sparse_251231.h5"
       era5: "210101_ERA5_sparse_251231.h5"
       gt: "sparse_targets_2021_2025.h5"
       std: "21_MeanStd_24.h5"
   ```

### 3. Dynamic Spatiotemporal Observations
| Dataset | Source Provider | Products / Variables | Expected Storage Path |
| :--- | :--- | :--- | :--- |
| **GEMS Satellite** | [NIER NESC](https://nesc.nier.go.kr/en/html/cntnts/73/satellite/introduction.do) | `AERAOD (V2.1)`, `O3T (V2.1)`, `NO2 (V3.0 before 2025/11/26, V4.0.1 after 2025/11/26)` | `<gems_dir>/L2_<Product>/yyyymm/dd/*ORI.nc` *(FTP recommended)* |
| **ERA5-Land** | [ECMWF CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download) ([API setup](https://cds.climate.copernicus.eu/how-to-api)) | Hourly surface reanalysis (see `era5_feature` in `config.yaml`) | `<era5_dir>/*.nc` *(via CDS API)* |
| **Ground Truth (TAQMN)** | [MOENV AirTW](https://airtw.moenv.gov.tw/CHT/Query/His_Data.aspx) | 75 national EPA stations (`PM2.5`, `PM10`, `O3`, `NO2`, `CO`, `SO2`) | `dataset_storage/raw/gt_data/yyyy/*.csv` |
| **Station Coordinates** | [MOENV Open Data](https://data.moenv.gov.tw/dataset/detail/AQX_P_07) | Geographic coordinates of 75 national stations | `dataset_storage/site_coordinate.csv` |

### 4. Static Geospatial & Emission Features
| Feature Layer | Source Provider | Resolution / Contents | Target File |
| :--- | :--- | :--- | :--- |
| **DSM Elevation** | [Data.gov.tw](https://data.gov.tw/dataset/7507) | 500m Digital Surface Model | `dataset_storage/raw/dsm_500m.h5` |
| **Land Use & Population** | [MOI SEGIS](https://segis.moi.gov.tw/STATCloud/QueryInterface) | 9 land-use categories & population density | `dataset_storage/raw/landuse_pop.h5` |
| **Emission Inventory** | [MOENV TEDS](https://air.moenv.gov.tw/envtopics/AirQuality_6.aspx) | TEDS 13.0 gridded emissions inventory | `dataset_storage/raw/TEDS13.h5` |

### 5. Independent External Validation Benchmarks (Optional)
| Benchmark Target | Data Portals / Descriptions | Station Metadata | Target File |
| :--- | :--- | :--- | :--- |
| **Local Environmental Protection Bureau (LEPB) Stations** | Consolidated 26 municipal stations across 4 cities:<br/>• Taipei: [Taipei DEP AQI](https://www.tldep.gov.taipei/Public/DownLoad/AqiHour.aspx)<br/>• Taoyuan: [Taoyuan DEP Portal](https://tydep.tycg.gov.tw/cp.aspx?n=20016)<br/>• Taichung: [Taichung TAQM](https://taqm.epb.taichung.gov.tw/)<br/>• Kaohsiung: [Kaohsiung Hourly Data](https://lab.ksepb.kcg.gov.tw/kaqm/tw/HourlyData.aspx) | • [Taipei Intro](https://www.tldep.gov.taipei/Public/EnvMonitoring/Air_AutoStation_Intro.aspx)<br/>• Taoyuan: in portal<br/>• [Taichung Info](https://taqm.epb.taichung.gov.tw/)<br/>• [Kaohsiung Coordinates](https://lab.ksepb.kcg.gov.tw/kaqm/tw/b0202.aspx) | `dataset_storage/processed/external_<yyyy>.h5`<br/>*(Consolidated 26 local stations)* |
| **TEPA Baseline** | [MOENV AirTW](https://airtw.moenv.gov.tw/) (IDW + Wind Speed + Terrain method) | Official comparative spatial baseline (point-extracted values at 101 station coordinates: 26 LEPB + 75 TAQMN) | `dataset_storage/processed/tepa_<yyyy>.h5` |

### 6. Configuration Setup
The repository is pre-configured out-of-the-box to execute immediately using the bundled **January 2025 Sample Dataset**.

When switching to raw data preprocessing or reproducing the full multi-year benchmarks:
1. Open `configs/config.yaml`.
2. Update the date ranges and processed file names as outlined in Section 2.B above.
3. If running raw data crawlers and preprocessing, update `paths: raw_data` (`gems_dir`, `gems_2025_dir`, `era5_dir`) to point to your local raw storage directories.
4. Verify that the corresponding processed tensors exist under `dataset_storage/processed/`.

## 🚀 Usage & Advanced Experiments

1. **Pipeline Modes & Customization:**
   * **Quickstart Execution**: Run out-of-the-box (`python main.py`) using the pre-configured sample dataset as shown in [Quickstart](#quickstart-installation--verification).
   * **Full Multi-Year Reproduction**: Update `configs/config.yaml` dates and filenames as detailed in [Dataset Section 2.B](#b-full-multi-year-dataset-20212025-benchmark-reproduction).
   *(Note: Data preprocessing will only generate `.h5` files if they are missing. It will never overwrite existing processed data).*

2. **Model Architecture:**
   The core deep learning model is **TR-STAUNet** (instantiated as `AttenUNet` / `AttenUNetTrainer` in `pipelines/trainer.py`), a 3D U-Net integrated with spatial-temporal attention mechanisms for capturing multi-scale atmospheric dispersion.

3. **Experiment Stages:**
   * **Model Training & Loss Optimization**: Trains with `CostSensitiveL1Loss` (which applies logarithmic concentration weights to emphasize severe pollution episodes and compensates scale differences for low-concentration pollutants like CO/SO2), physics-guided `transmission_resistance_loss` (incorporating ERA5 wind vectors and DSM terrain slope), and `temporal_consistency_loss` (for multi-hour prediction).
   * **Standard Multi-Step Evaluation (`trainer.evaluate()`)**: Evaluates predictions across test timestamps for each step $t$ (from 1 to `predict_step`) and caches output tensors into `<model>.h5`.
   * **Online Rolling Fine-Tuning (`trainer.evaluate_adaptive()`)**: Sequentially adapts the model to temporal distribution shifts on a daily basis using a rolling window of recent $K$ days (`rolling_days` in `config.yaml`).
   * **Feature & Temporal Ablation (`pipelines/ablation.py`)**: Runs systematic feature zeroing and temporal lookback ablations to assess the marginal contribution of each feature and lookback timestep.

4. **Run CatBoost Baseline:**
   In `main.py`, uncomment the `catboost.rolling_validation(train_loader, test_loader)` call to train and evaluate the gradient boosted decision tree baseline.

5. **Run Auxiliary & Diagnostic Tools (`scripts/`):**
   * **Data Profiling**: Compute multi-year spatial coverage and three-way comparative statistics across GEMS, ERA5, and ground stations (**requires the full multi-year archive `sparse_targets_2021_2025.h5`**; not applicable to the 1-month sample dataset):
     ```bash
     python scripts/data_profiler.py
     ```
   * **External Station Evaluation**: Validate model predictions against 26 independent municipal LEPB stations and the official TEPA baseline (execute post-training; configure `target_folder` inside `scripts/eval_external.py` to match your experiment directory name under `results/`, for example: `AttentionUnet_yyyymmdd`):
     ```bash
     python scripts/eval_external.py
     ```
   * **Projection Visualization**: Synthesize side-by-side video comparing raw scatter vs. interpolated grid features:
     ```bash
     python scripts/visualize_features.py
     ```

6. **Hyperparameter & Feature Configuration:**
   Modify training hyperparameters directly inside `configs/config.yaml`:
   * `lookback_window`: Number of preceding hours for input sequence (default: `3`).
   * `predict_step`: Number of forecast lead hours (default: `1`).
   * `rolling_days`: Prior days used for daily online fine-tuning (default: `1`).
   * `batch_size`: Training batch size (default: `4`; reduce to `2` or `1` if encountering VRAM / RAM constraints on smaller devices).
   * `learning_rate`, `fine_tune_learning_rate`, `epochs`.
   * `features.exclude`: Add or remove feature names to dynamically toggle active model input channels (e.g., excluding TEDS and land-use classes).

## 📊 Evaluation & Outputs

### Evaluation Metrics
Model performance is evaluated across 6 target air pollutants (`PM2.5`, `PM10`, `O3`, `NO2`, `CO`, `SO2`) using four standard metrics (`utils/metrics.py`):
* **$R^2$ Score (Coefficient of Determination)**: Measures proportion of variance explained by model predictions.
* **MAE (Mean Absolute Error)**: Quantifies average absolute prediction error magnitude.
* **RMSE (Root Mean Square Error)**: Measures error variance, heavily penalizing large discrepancies.
* **MAPE (Mean Absolute Percentage Error)**: Computes relative percentage discrepancy across non-zero ground truth.

Predictions are also benchmarked against the official **TEPA Baseline** (IDW + wind speed + terrain, evaluated at 101 station coordinates: 26 LEPB + 75 TAQMN) and validated against **26 independent municipal LEPB monitoring stations**.

### Artifacts & Results Structure
Outputs are automatically generated under `results/<Model>_<YYYYMMDD>/`:
* **Model Checkpoints**: Saved in `weights/` as `<model_name>.pth`.
* **Execution Logs (`results/.../logs/`)**:
  * `training_log.txt`: Epoch-by-epoch loss trajectories and timestamps.
  * `<model>_step<t>.txt`: Aggregate metrics ($R^2$, MAE, RMSE, MAPE) for each forecast step.
  * `<model>_rolling_<k>_step<t>.txt`: Aggregate metrics after online rolling fine-tuning.
  * `MAE_<model>.png`: Training and validation loss curves.
* **Visualizations (`results/.../step_<t>/` & `inference/`)**:
  * Spatial concentration heatmaps across Taiwan (`.png`).
  * Hourly and monthly error breakdown charts.
* **Prediction Cache**:
  * `<model>.h5`: Full spatiotemporal prediction tensor (`y_pred`, `timestamps`) for downstream benchmarking.
  * `config.json`: Complete snapshot of experiment configuration.

## 📖 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{tr_staunet_2026,
  title   = {Physics-Informed Deep Spatiotemporal Downscaling of GEMS Geostationary Satellite Observations for Short-Term Multi-Pollutant Forecasting},
  author  = {Huang, Tzu-Jui and Chan, Ta-Chien},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {Under review}
}
```

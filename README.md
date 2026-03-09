## Climate Trends of Tropical Cyclone Intensity and Energy Extremes Revealed by Deep Learning

This repository contains a PyTorch Lightning reimplementation of the tropical cyclone structure analysis model used in:

**Climate Trends of Tropical Cyclone Intensity and Energy Extremes Revealed by Deep Learning**  
Buo-Fu Chen, Center for Weather Climate and Disaster Research, National Taiwan University, Taiwan.

The code trains a profiler network on the TCSA dataset and provides scripts for training, evaluation, offline prediction, and ONNX export.


## Installation

This project targets **Python 3.12+** and is packaged with `pyproject.toml`.

- **Clone the repository**:

```bash
git clone <this-repo-url>
cd dsat-lightning
```

- **Using `uv` (recommended)**:

```bash
uv sync
```

## Data setup (TCSA)

The model expects the TCSA dataset in HDF5 format (`TCSA.h5`), which is then preprocessed into yearly pickle files on first use.

- **1. Obtain the TCSA dataset** (`TCSA.h5`) from the original authors or your local copy.  
- **2. Choose a data directory**, for example:

```bash
export DATA_DIR=/path/to/TCSA_data_2004_2018
mkdir -p "$DATA_DIR"
cp /where/you/have/TCSA.h5 "$DATA_DIR"/
```

- **3. Point Hydra to the project and data roots** (recommended for running from the repo root):

```bash
export PROJECT_ROOT=$(pwd)
export DATA_DIR=/path/to/TCSA_data_2004_2018
```

On the first run, the `TCSAPreprocessor` will read `TCSA.h5`, generate yearly pickles (`TCSA.<year>.polar.pickle`), and cache them under `${DATA_DIR}`. This can take some time and requires tens of GB of disk space.


## Training

Training and evaluation are driven by **Hydra** configs under `configs/`.

- **Default training run** (from the repo root):

```bash
python -m dsat_lightning.train
```

By default this uses `configs/train.yaml`, which in turn composes:

- `data: tcsa_polar` (`configs/data/tcsa_polar.yaml`)  
- `model: profiler_ir1`  
- `callbacks: default`  
- `paths: default` (uses `PROJECT_ROOT` and `DATA_DIR`)  
- `logger: wandb`

- **Common overrides** (Hydra syntax):

```bash
# Change max epochs and batch size
python -m dsat_lightning.train trainer.max_epochs=100 data.batch_size=256

# Disable test-after-train
python -m dsat_lightning.train test=false
```

Checkpoints, logs, and Hydra run directories are written under `logs/` by default (see `lightning-hydra-template/configs/hydra/default.yaml` and `configs/paths/default.yaml`).


## Offline prediction

To run the trained model on the test set and save predicted/ground-truth profiles:

```bash
python -m dsat_lightning.predict ckpt_path=/path/to/model.ckpt \
  predict.output_file=profiles.pt
```

Notes:

- `ckpt_path` is **required**; it can also be provided via the `CKPT_PATH` environment variable:

```bash
CKPT_PATH=/path/to/epoch=29-step=2280.ckpt python -m dsat_lightning.predict
```

- Outputs are saved to `${paths.output_dir}/profiles.pt` by default, where `paths.output_dir` comes from Hydra (typically under `logs/`).


## Exporting to ONNX

Use the helper script in `scripts/` to export a Lightning checkpoint to ONNX:

```bash
python scripts/export_onnx.py ckpt_path=/path/to/model.ckpt
```

Optional arguments (Hydra overrides):

```bash
python scripts/export_onnx.py ckpt_path=/path/to/model.ckpt \
  export.output_file=model.onnx \
  export.image_height=180 export.image_width=103 \
  export.batch_size=1
```

You can also pass the checkpoint path via environment variable (useful when the filename contains `=`):

```bash
CKPT_PATH=/path/to/epoch=29-step=2280.ckpt python scripts/export_onnx.py
```

The exported ONNX model takes an image (`image`, shape \(B, C, H, W\)) and an auxiliary feature vector (`feature`, shape \(B, 9\)) and returns a wind profile (`profile`, shape \(B, 151\)).


## Project structure (high level)

- `src/dsat_lightning/` – package root
  - `dataset/` – TCSA dataset preprocessing and `LightningDataModule`
  - `model/` – profiler model and components
  - `utils/` – logging, Hydra helpers, and training utilities
- `configs/` – Hydra configuration for data, model, paths, and training/prediction
- `scripts/export_onnx.py` – export trained model to ONNX
- `outputs/`, `logs/`, `wandb/` – run artifacts (ignored by git)
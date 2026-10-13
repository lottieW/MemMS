# _MemMS: Memory-Guided Multi-Scale Modeling for Time Series Anomaly Detection_

This repository provides the implementation of _MemMS: Memory-Guided Multi-Scale Modeling for Time Series Anomaly Detection_, referred to as _MemMS_ below.

The implementation uses the Merlion libraries and supports **SMD, Exathlon, SWaT, and ESA**.

## Installation
This code is based on `Python 3.12` and `PyTorch 2.6`. Additionally, we should install `saleforce-merlion v2.0.0` and `ts_dataset` as Merlion suggested.

```
pip install salesforce-merlion==2.0.0
```
This repository already includes the merlion's data loading package `ts_datasets`.
Download the data and unzip it in `data/SMD`.


## Repository Structure

### `conf`
This directory contains experiment parameters for the MOC model on IOpsCompetition, SWaT, and WADI datasets.

### `models`
Source code of the MOC models.


## Usage

Run commands from the repository root:

```bash
# MemMS on SMD
python MemMS.py --selected_dataset SMD --device cuda --seed 1

# MemMS on Exathlon
python MemMS.py --selected_dataset Exathlon --device cuda --seed 1

# MemMS on SWaT
python MemMS.py --selected_dataset SWaT --device cuda --seed 1

# MemMS on ESA
python MemMS.py --selected_dataset ESA --device cuda --seed 1
```

Use `--device cpu` to run on a CPU. Hyperparameters are loaded automatically from `conf/memms/<dataset>_Configs.py`.


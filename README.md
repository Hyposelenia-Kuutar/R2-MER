# R2-MER

PyTorch implementation of **R2-MER: Micro-Expression Recognition via Region-Aware Reconstruction and Modality-Aware Fusion**.

R2-MER is a two-stage micro-expression recognition framework consisting of:

1. **Region-aware reconstruction pre-training**, which learns transferable facial representations from region-swapped Apex images.
2. **Supervised micro-expression recognition**, which combines an RGB image branch and an optical-flow branch through modality-aware feature fusion.

The supervised stage is evaluated using **leave-one-subject-out (LOSO) cross-validation**.

---

## Project Structure

```text
R2-MER/
├── configs/
│   ├── __init__.py
│   ├── pretrain.py
│   ├── casme2_3class.py
│   ├── casme2_5class.py
│   ├── samm_3class.py
│   ├── samm_5class.py
│   ├── smic_3class.py
│   └── three_db.py
│
├── data/
│   └── annotations/
│       ├── casme2_3class.csv
│       ├── casme2_5class.csv
│       ├── samm_3class.csv
│       ├── samm_5class.csv
│       └── smic_3class.csv
│
├── datasets/
│   ├── __init__.py
│   ├── base.py
│   ├── casme2.py
│   ├── samm.py
│   ├── smic.py
│   └── legacy.py
│
├── models/
│   ├── __init__.py
│   ├── rgb_branch.py
│   ├── htnet.py
│   └── r2mer.py
│
├── tools/
│   ├── generate_mix.py
│   ├── pretrain.py
│   ├── train.py
│   └── run_loso.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

The main directories are organized as follows:

- `configs/`: dataset paths, GPU settings, training settings, and hyperparameter grids.
- `data/annotations/`: annotation CSV files used by the experiments.
- `datasets/`: dataset loading and preprocessing for CASME II, SAMM, and SMIC.
- `models/`: RGB branch, HTNet optical-flow branch, and the complete R2-MER model.
- `tools/`: mixed-image generation, pre-training, supervised training, and LOSO evaluation.

---

## Environment

The experiments were implemented using:

- Linux (Ubuntu 22.04)
- Python 3.9
- PyTorch 2.0.1
- NVIDIA GPU with CUDA 12.2
- Conda (Miniconda or Anaconda)

Install the required packages with:

```bash
pip install -r requirements.txt
```

It is recommended to create an isolated Python environment before installation.

For example:

```bash
conda create -n r2mer_env python=3.9
conda activate r2mer_env

pip install -r requirements.txt
```

---

## Dataset Preparation

The experiments are conducted on three commonly used micro-expression datasets:

- **CASME II**
- **SAMM**
- **SMIC**

### 1. Obtain the Original Datasets

The original datasets should be requested or downloaded according to the licenses and access requirements of their official websites:

- **CASME II**:  
  [http://casme.psych.ac.cn/casme/e2](http://casme.psych.ac.cn/casme/e2)

- **SAMM**:  
  [http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)

- **SMIC**:  
  [https://www.oulu.fi/en/university/faculties-and-units/faculty-information-technology-and-electrical-engineering/center-for-machine-vision-and-signal-analysis](https://www.oulu.fi/en/university/faculties-and-units/faculty-information-technology-and-electrical-engineering/center-for-machine-vision-and-signal-analysis)


### 2. Prepare RGB Frames and Optical Flow

After obtaining the original datasets, prepare the RGB frame sequences and extract the optical flow between the **Onset** and **Apex** frames.

The current implementation expects the optical-flow input to be stored as a three-channel image with the filename:

```text
flow_{onset}_{apex}.png
```

For example:

```text
flow_61_94.png
```

Optical-flow maps must be generated before supervised training.

Alternatively, the dataset and preprocessing resources provided by the **HTNet** project can be used:

[https://github.com/wangzhifengharrison/HTNet](https://github.com/wangzhifengharrison/HTNet)

However, regardless of whether the data are prepared from the original datasets or obtained from HTNet, the RGB frames and optical-flow maps must be reorganized according to the directory structures expected by this repository.

---

## Expected Dataset Structure

### CASME II

The CASME II RGB frames should follow:

```text
CASME2_RGB/
├── sub01/
│   ├── EP02_01f/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   ├── ...
│   │   └── imgN.jpg
│   └── ...
├── sub02/
│   └── ...
└── ...
```

The corresponding optical-flow maps should follow the same subject and sequence hierarchy:

```text
CASME2_FLOW/
├── sub01/
│   ├── EP02_01f/
│   │   └── flow_{onset}_{apex}.png
│   └── ...
├── sub02/
│   └── ...
└── ...
```

For example:

```text
CASME2_RGB/
└── sub01/
    └── EP02_01f/
        ├── img61.jpg
        └── img94.jpg

CASME2_FLOW/
└── sub01/
    └── EP02_01f/
        └── flow_61_94.png
```

The configuration should then specify:

```python
"data": {
    "csv_path": "data/annotations/casme2_3class.csv",
    "rgb_root": "/path/to/CASME2_RGB",
    "flow_root": "/path/to/CASME2_FLOW",
}
```

---

### SAMM

The SAMM RGB frames should be organized as:

```text
SAMM_RGB/
├── 006/
│   ├── <sequence_name>/
│   │   ├── 006_00001.jpg
│   │   ├── 006_00002.jpg
│   │   ├── ...
│   │   └── 006_XXXXX.jpg
│   └── ...
├── 007/
│   └── ...
└── ...
```

The current loader accepts common SAMM frame-number formats such as:

```text
006_0001.jpg
006_00001.jpg
006_1.jpg
```

The optical-flow maps should be organized as:

```text
SAMM_FLOW/
├── 006/
│   ├── <sequence_name>/
│   │   └── flow_{onset}_{apex}.png
│   └── ...
├── 007/
│   └── ...
└── ...
```

For example:

```text
SAMM_RGB/
└── 006/
    └── 006_1_2/
        ├── 006_00123.jpg
        └── 006_00157.jpg

SAMM_FLOW/
└── 006/
    └── 006_1_2/
        └── flow_123_157.png
```

The configuration should then specify:

```python
"data": {
    "csv_path": "data/annotations/samm_3class.csv",
    "rgb_root": "/path/to/SAMM_RGB",
    "flow_root": "/path/to/SAMM_FLOW",
}
```

---

### SMIC

The SMIC RGB frames should be organized according to subject and clip:

```text
SMIC_RGB/
├── <subject>/
│   ├── <clip_name>/
│   │   ├── reg_image1.bmp
│   │   ├── reg_image2.bmp
│   │   ├── ...
│   │   └── reg_imageN.bmp
│   └── ...
└── ...
```

Frame names such as the following are supported:

```text
reg_image1.bmp
reg_image01.bmp
reg_image001.bmp
reg_image0001.bmp
```

The optical-flow maps should be organized as:

```text
SMIC_FLOW/
├── <subject>/
│   ├── <clip_name>/
│   │   └── flow_{onset}_{apex}.png
│   └── ...
└── ...
```

For example:

```text
SMIC_RGB/
└── 1/
    └── micro_negative_01/
        ├── reg_image1.bmp
        ├── ...
        └── reg_image12.bmp

SMIC_FLOW/
└── 1/
    └── micro_negative_01/
        └── flow_1_12.png
```

The subject and clip directory names must match the corresponding entries in the annotation CSV file.

The configuration should then specify:

```python
"data": {
    "csv_path": "data/annotations/smic_3class.csv",
    "rgb_root": "/path/to/SMIC_RGB",
    "flow_root": "/path/to/SMIC_FLOW",
}
```

---

## Annotation Files

The annotation files used by the code are provided under:

```text
data/annotations/
```

The current repository supports:

```text
casme2_3class.csv
casme2_5class.csv
samm_3class.csv
samm_5class.csv
smic_3class.csv
```

Please keep the annotation files and dataset directory names consistent. The subject ID, sequence name, Onset frame, and Apex frame recorded in the CSV files are used to locate the corresponding RGB and optical-flow inputs.

---

## Stage I: Reconstruction Pre-training

Before supervised training, R2-MER first performs region-aware reconstruction pre-training.

The pre-training stage uses Apex frames from CASME II, SAMM, and SMIC to construct region-swapped facial images. The mixed image and its corresponding original Apex image are then used as the input-target pair for reconstruction learning.

### 1. Configure Pre-training

Edit:

```text
configs/pretrain.py
```

Example:

```python
CONFIG = {
    "datasets": ["casme2", "smic", "samm"],

    "data": {
        "casme2": {
            "rgb_root": "/path/to/CASME2_RGB",
            "csv_path": "data/annotations/casme2_3class.csv",
        },
        "smic": {
            "rgb_root": "/path/to/SMIC_RGB",
            "csv_path": "data/annotations/smic_3class.csv",
        },
        "samm": {
            "rgb_root": "/path/to/SAMM_RGB",
            "csv_path": "data/annotations/samm_3class.csv",
        },
    },

    "mix_save_base": "/path/to/mixed_images",
    "save_path": "/path/to/pretrain.pth",

    "gpu_ids": "0,1",

    "epochs": 100,
    "batch_size": 128,

    "optimizer": {
        "lr": 5e-5,
        "weight_decay": 0.01,
    },

    "loss": {
        "lambda_l1": 0.6,
        "lambda_ssim": 0.4,
    },

    "grad_clip": 1.0,
}
```

### 2. Run Pre-training

From the project root directory:

```bash
python -u tools/pretrain.py --config configs/pretrain.py
```

or simply:

```bash
python -u tools/pretrain.py
```

The region-swapped images will be generated automatically if the corresponding mixed-image directory does not already contain generated samples.

The pre-training objective is:

```text
L_pre = 0.6 * L1 + 0.4 * (1 - SSIM)
```

The latest checkpoint is continuously updated according to the path specified by:

```python
"save_path"
```

For example:

```text
pretrain.pth
```

Epoch-specific checkpoints can also be retained using filenames containing the epoch number and reconstruction loss.

---

## Stage II: Supervised Training

After reconstruction pre-training, specify the generated checkpoint in each supervised configuration file:

```python
"pretrained_weights": "/path/to/pretrain.pth"
```

R2-MER is then trained and evaluated using LOSO cross-validation.

The supervised model consists of:

- a pre-trained RGB image branch,
- an HTNet-based optical-flow branch,
- a modality-aware feature fusion module,
- and a final classification head.

---

## Hyperparameter Configuration

Hyperparameter combinations are specified directly in each dataset configuration file.

For example:

```python
"param_grid": {
    "learning_rate": [5e-5],
    "weight_decay": [1e-4],
    "batch_size": [128],
    "epochs": [500],
    "num_classes": [3],
    "seed": [42],
}
```

Multiple values can be provided:

```python
"param_grid": {
    "learning_rate": [5e-5, 1e-5],
    "weight_decay": [1e-4, 1e-3],
    "batch_size": [64, 128],
    "epochs": [500],
    "num_classes": [3],
    "seed": [42],
}
```

`run_loso.py` automatically constructs and evaluates all hyperparameter combinations.

For example:

```text
2 learning rates
× 2 weight decays
× 2 batch sizes
= 8 hyperparameter combinations
```

Each hyperparameter combination is evaluated over the complete LOSO protocol.

---

## CASME II Three-Class Evaluation

```bash
python -u tools/run_loso.py --config configs/casme2_3class.py
```

---

## SAMM Three-Class Evaluation

```bash
python -u tools/run_loso.py --config configs/samm_3class.py
```

---

## SMIC Three-Class Evaluation

```bash
python -u tools/run_loso.py --config configs/smic_3class.py
```

---

## CASME II Five-Class Evaluation

```bash
python -u tools/run_loso.py --config configs/casme2_5class.py
```

---

## SAMM Five-Class Evaluation

```bash
python -u tools/run_loso.py --config configs/samm_5class.py
```

---

## Joint Three-Dataset Evaluation

The joint CASME II + SAMM + SMIC experiment is configured through:

```text
configs/three_db.py
```

Run:

```bash
python -u tools/run_loso.py --config configs/three_db.py
```

---

## GPU Configuration

GPU selection is controlled by each configuration file.

For supervised training:

```python
"gpu_id": "0"
```

Different datasets can therefore be assigned to different GPUs.

For example:

```text
CASME II -> GPU 0
SAMM     -> GPU 1
SMIC     -> GPU 2
3DB      -> GPU 3
```

---

## Running Experiments in the Background

For long-running experiments on a remote server, `nohup` can be used.

### CASME II

```bash
nohup python -u tools/run_loso.py \
    --config configs/casme2_3class.py \
    > casme2_3class.log 2>&1 &
```

### SAMM

```bash
nohup python -u tools/run_loso.py \
    --config configs/samm_3class.py \
    > samm_3class.log 2>&1 &
```

### SMIC

```bash
nohup python -u tools/run_loso.py \
    --config configs/smic_3class.py \
    > smic_3class.log 2>&1 &
```

### Joint Three-Dataset Setting

```bash
nohup python -u tools/run_loso.py \
    --config configs/three_db.py \
    > three_db.log 2>&1 &
```

### The training log can be monitored using:

```bash
tail -f casme2_3class.log
```

### GPU usage can be monitored using:

```bash
nvidia-smi
```

---

## Output

The output directory is controlled by:

```python
"output_root"
```

in each configuration file.

For each hyperparameter setting, `run_loso.py` automatically creates a separate experiment directory.

The summary CSV records information including:

```text
Experiment_ID
Learning Rate
Weight Decay
Batch Size
Epochs
Seed
Overall UF1
Overall UAR
Overall Accuracy
Running Time
```

---

## Evaluation Metrics

The main evaluation metrics are:

- **UF1**: Unweighted F1-score
- **UAR**: Unweighted Average Recall

The experiments are evaluated under the LOSO protocol commonly adopted in micro-expression recognition.

---

## Notes

- The original CASME II, SAMM, and SMIC datasets are not redistributed in this repository.
- Please request the original datasets from their respective official websites and follow their licenses and terms of use.
- After obtaining the original datasets, RGB frames and Onset-to-Apex optical-flow maps must be prepared before supervised training.
- The preprocessed dataset resources provided by HTNet can also be used, but they must be reorganized according to the directory structures and filename conventions required by this repository.
- The annotation CSV files under `data/annotations/` determine the subject, sequence, Onset, and Apex information used by the dataset loaders.
- Dataset paths, GPU IDs, output directories, and hyperparameters should be configured in `configs/*.py`.
- Reconstruction pre-training should be completed before supervised recognition training.
- All commands should preferably be executed from the project root directory.

---

## Acknowledgments

This implementation was developed based in part on the code framework of **HTNet**:

[https://github.com/wangzhifengharrison/HTNet](https://github.com/wangzhifengharrison/HTNet)

We sincerely thank the authors of HTNet for making their implementation and related resources publicly available.

In particular, the overall code framework of this repository and the implementation of the optical-flow branch were developed based on and adapted from the HTNet codebase. Their open-source implementation provided an important foundation for the development and implementation of R2-MER.

The dataset and preprocessing resources released with HTNet can also be used to prepare the inputs required by this repository. When using these resources, please reorganize the RGB frames and optical-flow maps according to the dataset structures described above.

If you use this repository, please also consider citing the original HTNet work.

---


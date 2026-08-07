# R2-MER: Micro-Expression Recognition via Region-Aware Reconstruction and Modality-Aware Fusion

Official PyTorch implementation of **R2-MER**, a two-stage dual-stream framework for **Micro-Expression Recognition (MER)**.

R2-MER first learns region-aware facial representations through a facial key-region perturbation and reconstruction proxy task. During downstream recognition, the pretrained RGB branch models static appearance and inter-region relationships, while HTNet extracts dynamic motion representations from regional optical flow. The two modalities are then integrated by a Modality-Aware Fusion Module for micro-expression classification.

---

## ✨ Highlights

* **Facial Key-Region Perturbation Pre-training**
  Constructs perturbed Apex images by exchanging four facial regions around the two eyes and two mouth corners. The model learns to recover the original facial appearance without requiring emotion labels.

* **Multi-Dataset Pre-training**
  Apex frames from multiple datasets can be merged into a unified pre-training pool. Region exchange can therefore occur both within and across datasets.

* **Holistic Key-Primed Feature Extractor (HKPFE)**
  Employs parallel high- and low-resolution branches to preserve fine-grained local information while incorporating broader contextual cues.

* **Inter-Region Synergy Attention (IRSA)**
  Models interactions between the perturbed and original facial representations using cross-attention, lightweight positional encoding, feature expansion, and gated residual fusion.

* **Modality-Aware Fusion Module (MAFM)**
  Integrates static RGB representations with dynamic optical-flow representations through cross-modal feature mapping, interaction gating, and channel-aware fusion.

* **LOSO Evaluation Pipeline**
  Provides automatic Leave-One-Subject-Out evaluation and reports Accuracy, Unweighted F1-score (UF1), Unweighted Average Recall (UAR), and confusion matrices.

---

## 🧠 Framework

R2-MER consists of two stages.

### Stage I: Region-Aware Reconstruction Pre-training

Apex frames are first collected from one or multiple datasets. MTCNN is used to locate five facial landmarks, and four key regions corresponding to the two eyes and two mouth corners are selected for perturbation.

For each target Apex image, the corresponding regions from another Apex image are substituted to construct a mixed image.

```text
CASME II Apex ─┐
SAMM Apex ─────┼──> Unified Apex Pool
SMIC Apex ─────┘
                       │
                       ▼
                MTCNN Landmarks
                       │
                       ▼
           Eye / Mouth Region Exchange
                       │
                       ▼
                 Mixed Apex Image

        Original Apex          Mixed Apex
             │                     │
             ▼                     ▼
          HKPFE                  HKPFE
             │                     │
             └──────── IRSA ───────┘
                       │
                       ▼
                    Decoder
                       │
                       ▼
              Reconstructed Apex
```

The reconstruction objective is

```text
L_pre = 0.6 × L1 + 0.4 × (1 - SSIM)
```

The resulting pretrained RGB encoder and attention modules are transferred to the downstream MER task.

---

### Stage II: Dual-Stream Micro-Expression Recognition

The downstream recognition framework contains an RGB stream and an optical-flow stream.

```text
        Onset RGB ───────────┐
                             │
                             ▼
                           HKPFE
                             │
                             │
        Apex RGB ────────────┤
                             ▼
                            IRSA
                             │
                     Static RGB Feature
                             │
                             │
                             ▼
                         ┌── MAFM ──┐
                         │          │
                         │          ▼
                         │      Classifier
                         │          │
                         │          ▼
                         │     Prediction
                         │
Regional Optical Flow    │
        │                 │
        ▼                 │
      HTNet ──────────────┘
        │
 Dynamic Flow Feature
```

The RGB branch remains frozen during supervised training. HTNet is initially frozen and is subsequently unfrozen for joint optimization with the fusion module.

---

## 📁 Repository Structure

```text
R2-MER/
├── main.py
├── train.py
├── dataset.py
├── Model.py
├── RGB_model.py
├── generate_mix.py
├── pre_train.py
├── requirements.txt
└── README.md
```

| File               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `main.py`          | Full LOSO experiment launcher and result aggregation         |
| `train.py`         | Training and evaluation for a single LOSO fold               |
| `dataset.py`       | CASME II, SAMM, and SMIC dataset loaders                     |
| `Model.py`         | HTNet, MAFM, and the final dual-stream recognition model     |
| `RGB_model.py`     | HKPFE, IRSA, Decoder, and RegionRecoveryModel                |
| `generate_mix.py`  | Multi-dataset Apex collection and facial region perturbation |
| `pre_train.py`     | Region-aware reconstruction pre-training                     |
| `requirements.txt` | Python dependencies                                          |

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/R2-MER.git
cd R2-MER
```

### 2. Create an environment

```bash
conda create -n r2mer python=3.8
conda activate r2mer
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The pre-training and data-processing code additionally requires:

```bash
pip install pytorch-msssim mediapipe tqdm
```

The main environment used by the current implementation includes:

```text
PyTorch       1.12.1
torchvision   0.13.1
NumPy         1.23.1
OpenCV        4.6.0
Pandas        1.4.4
scikit-learn  1.1.2
facenet-pytorch 2.5.2
einops        0.4.1
```

A CUDA-enabled GPU is recommended.

---

## 📂 Dataset Preparation

The current recognition pipeline supports:

* **CASME II**
* **SAMM**
* **SMIC**

The datasets themselves are not included in this repository. Please obtain them from their official sources and follow their respective licenses.

For downstream recognition, three types of information are required:

1. RGB facial frames;
2. onset/apex annotations and class labels;
3. pre-computed onset-to-apex optical-flow images.

The classification label column is automatically selected according to:

```python
label_col = f"{num_classes}label"
```

For example:

```text
3label
5label
```

can be used for three-class and five-class experiments, respectively, provided that the corresponding labels are included in the CSV annotation file.

---

## Data Organization

### CASME II

Example RGB structure:

```text
casme2_256/
└── sub01/
    └── EP02_01f/
        ├── img1.jpg
        ├── img2.jpg
        └── ...
```

Example optical-flow structure:

```text
CASME2_FLOW/
└── sub01/
    └── EP02_01f/
        └── flow_<onset>_<apex>.png
```

The CSV file should contain fields such as:

```text
Subject
Filename
OnsetFrame
ApexFrame
3label
```

---

### SAMM

Example RGB structure:

```text
samm_256/
└── 006/
    └── <sequence>/
        ├── 006_0001.jpg
        ├── 006_0002.jpg
        └── ...
```

Example optical-flow structure:

```text
SAMM_FLOW/
└── 006/
    └── <sequence>/
        └── flow_<onset>_<apex>.png
```

The annotations should include fields such as:

```text
Subject
Filename
Onset Frame
Apex Frame
3label
```

---

### SMIC

Example RGB structure:

```text
smic_256/
└── s01/
    └── <clip>/
        ├── reg_image1.bmp
        ├── reg_image2.bmp
        └── ...
```

Example optical-flow structure:

```text
SMIC_FLOW/
└── s01/
    └── <clip>/
        └── flow_<onset>_<apex>.png
```

The annotation file should contain fields such as:

```text
subject
clip
onset_frame
apex_frame
3label
```

---

# 🚀 Usage

## 1. Region-Aware Reconstruction Pre-training

`pre_train.py` supports both single-dataset and multi-dataset pre-training.

### Multi-Dataset Pre-training

For example, CASME II, SMIC, and SAMM can be treated as one unified Apex pool:

```bash
python pre_train.py \
    --datasets casme2 smic samm \
    --orig_data_roots \
        /path/to/casme2 \
        /path/to/smic \
        /path/to/samm \
    --csv_paths \
        /path/to/casme2.csv \
        /path/to/smic.csv \
        /path/to/samm.csv \
    --mix_save_base /path/to/multi_exchange_pool \
    --save_path /path/to/pretrain.pth \
    --batch_size 128 \
    --epochs 100
```

The order of the three arguments must correspond exactly:

```text
--datasets
--orig_data_roots
--csv_paths
```

For example:

```text
casme2 -> /path/to/casme2 -> /path/to/casme2.csv
smic   -> /path/to/smic   -> /path/to/smic.csv
samm   -> /path/to/samm   -> /path/to/samm.csv
```

---

### Single-Dataset Pre-training

For SMIC only:

```bash
python pre_train.py \
    --datasets smic \
    --orig_data_roots /path/to/smic \
    --csv_paths /path/to/smic.csv \
    --save_path /path/to/pretrain_smic.pth
```

---

## Cross-Dataset Region Perturbation

All valid Apex images from the selected datasets are merged before region exchange.

For example:

```text
Target: CASME II
Source: SMIC

CASME II Apex
      +
SMIC eye / mouth regions
      ↓
Mixed Apex
```

Similarly, the implementation allows:

```text
CASME II <- CASME II
CASME II <- SAMM
CASME II <- SMIC

SAMM     <- CASME II
SAMM     <- SAMM
SAMM     <- SMIC

SMIC     <- CASME II
SMIC     <- SAMM
SMIC     <- SMIC
```

If `N` Apex images successfully pass landmark detection, the current implementation generates up to approximately:

```text
N × N
```

target-source combinations.

Because this Cartesian-product strategy can generate a large number of images, sufficient disk space is required.

---

## Pre-training Details

Each image is resized to:

```text
224 × 224
```

The default facial patch size used for region exchange is:

```text
56 × 56
```

The four exchanged landmark regions are:

```text
left eye
right eye
left mouth corner
right mouth corner
```

The nose landmark is not exchanged.

Default optimization settings are:

| Setting           |            Value |
| ----------------- | ---------------: |
| Epochs            |              100 |
| Batch size        |              128 |
| Optimizer         |            AdamW |
| Learning rate     |             5e-5 |
| Weight decay      |             0.01 |
| LR scheduler      | Cosine Annealing |
| L1 weight         |              0.6 |
| SSIM weight       |              0.4 |
| Gradient clipping |              1.0 |

The reconstruction loss is:

```text
L_pre = 0.6 × L1 + 0.4 × (1 - SSIM)
```

The latest checkpoint is saved after each epoch and contains:

```python
{
    "epoch": ...,
    "model_state_dict": ...,
    "loss": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...
}
```

---

# 2. Supervised MER Training

A single LOSO fold can be trained using `train.py`.

Example for CASME II:

```bash
python train.py \
    --dataset casme2 \
    --loso_subject 1 \
    --preds_save_path ./results/1_preds.pt \
    --weights_save_dir ./weights \
    --csv_path /path/to/3casme2.csv \
    --data_root /path/to/casme2_256 \
    --flow_root /path/to/CASME2_FLOW \
    --recovery_weights /path/to/pretrain.pth \
    --num_classes 3 \
    --batch_size 128 \
    --epochs 500 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --gpu 0 \
    --seed 42
```

Supported values of `--dataset` are:

```text
casme2
samm
smic
3db
```

---

## Recognition Input

Each downstream sample contains:

```text
x1: regional onset-to-apex optical flow
x2: onset RGB image
x3: apex RGB image
```

The onset and Apex RGB images are resized to:

```text
3 × 224 × 224
```

For the optical-flow branch, the full onset-to-apex flow image is resized to `28 × 28`. Four local `14 × 14` regions are then extracted according to facial landmarks:

```text
Left Eye        Right Eye
    ┌──────────────┐
    │              │
    └──────────────┘

Left Mouth      Right Mouth
    ┌──────────────┐
    │              │
    └──────────────┘
```

The four regions are concatenated into a final:

```text
3 × 28 × 28
```

regional optical-flow representation for HTNet.

---

## Recognition Model

### RGB Stream

The RGB branch is implemented by `RegionRecoveryModel`.

It contains:

```text
Onset RGB ──> HKPFE ──┐
                      │
                      ├── IRSA ──> Static RGB Feature
                      │
Apex RGB ───> HKPFE ──┘
```

During pre-training, the IRSA feature is sent to the Decoder for image reconstruction.

During downstream recognition, the Decoder is bypassed and the attended feature is projected to 1024 channels followed by global average pooling.

---

### Optical-Flow Stream

The dynamic branch is based on HTNet.

The current default configuration is:

```python
htnet_config = {
    "image_size": 28,
    "patch_size": 7,
    "num_classes": 3,
    "dim": 256,
    "heads": 3,
    "num_hierarchies": 3,
    "block_repeats": (2, 2, 8)
}
```

HTNet extracts hierarchical representations from the regional onset-to-apex optical-flow input.

---

### Modality-Aware Fusion Module

The Modality-Aware Fusion Module is implemented as `ModalityAwareFusion` in `Model.py`.

Given RGB and optical-flow representations, MAFM performs:

```text
RGB Feature ──> Normalization ──> RGB-to-Flow Mapping ──┐
                                                       │
                                                       ├─> Interaction
                                                       │    Gating
                                                       │       +
                                                       │ Channel Attention
                                                       │       │
Flow Feature ─> Normalization ──> Flow-to-RGB Mapping ─┘       │
                                                               ▼
                                                         Feature Fusion
                                                               │
                                                               ▼
                                                        Classification
```

The fused feature is finally passed to the HTNet classification head.

---

# 3. LOSO Evaluation

For a dataset containing `S` subjects, Leave-One-Subject-Out evaluation repeatedly uses:

```text
Training set = S - 1 subjects
Test set     = 1 held-out subject
```

`main.py` automatically obtains the subject list from the CSV annotation file and launches `train.py` for every subject.

Before running:

```bash
python main.py
```

configure the following paths near the beginning of `main.py`:

```python
BASE_ROOT = "/path/to/data"

CSV_FILE_PATH = "/path/to/annotations.csv"
DATA_ROOT = "/path/to/RGB"
FLOW_ROOT = "/path/to/optical_flow"

SAVE_ROOT = "/path/to/results"

UNIVERSAL_WEIGHT_PATH = "/path/to/pretrain.pth"

GPU_ID = "0"
```

Then configure the hyperparameters:

```python
PARAM_GRID = {
    "learning_rate": [5e-5],
    "weight_decay": [1e-4],
    "batch_size": [128],
    "epochs": [500],
    "dataset": ["casme2"],
    "num_classes": [3],
    "seed": [42]
}
```

Run:

```bash
python main.py
```

---

## Hyperparameter Search

`main.py` supports Cartesian-product hyperparameter search.

For example:

```python
PARAM_GRID = {
    "learning_rate": [1e-5, 5e-5],
    "weight_decay": [1e-4, 1e-2],
    "batch_size": [64, 128],
    "epochs": [300, 500],
    "dataset": ["casme2"],
    "num_classes": [3],
    "seed": [42, 2026]
}
```

Each configuration is evaluated using the full LOSO protocol.

---

## Training Strategy

The pretrained RGB branch is frozen throughout supervised training.

HTNet is initially frozen:

```text
Epoch 1–99:
    RGB branch     Frozen
    HTNet          Frozen
    MAFM           Trainable
```

At epoch 100:

```text
Epoch 100+:
    RGB branch     Frozen
    HTNet          Trainable
    MAFM           Trainable
```

The supervised objective is standard cross-entropy loss optimized with AdamW.

> **Implementation note:** in the current code, the HTNet classification head is part of `model.htnet`, so it is also frozen before epoch 100.

---

# 4. Three-Dataset Joint Training

`train.py` additionally provides a `3db` mode for joint training on:

```text
CASME II + SAMM + SMIC
```

For a target subject from one dataset, the corresponding subject is excluded only from that target dataset, while the other two datasets are included in the training set.

For example:

```text
Target: CASME II Subject 01

Training:
    CASME II except Subject 01
    + all SAMM samples
    + all SMIC samples

Testing:
    CASME II Subject 01
```

The LOSO subject identifier follows:

```text
<dataset>_<subject>
```

for example:

```text
casme2_01
samm_006
smic_s01
```

An example command is:

```bash
python train.py \
    --dataset 3db \
    --loso_subject casme2_01 \
    --preds_save_path ./results/casme2_01_preds.pt \
    --recovery_weights ./pretrain.pth \
    --num_classes 3 \
    --batch_size 128 \
    --epochs 500 \
    --lr 5e-5 \
    --gpu 0
```

The dataset paths used in `3db` mode are currently configured directly inside `train.py`.

---

## 📊 Evaluation Metrics

The complete LOSO pipeline reports:

* **Accuracy (ACC)**
* **Unweighted F1-score (UF1)**
* **Unweighted Average Recall (UAR)**
* **Confusion Matrix**

UF1 is computed as macro F1:

```python
f1_score(
    targets,
    predictions,
    average="macro"
)
```

UAR is computed as macro recall:

```python
recall_score(
    targets,
    predictions,
    average="macro"
)
```

---

## 📈 Output

For each hyperparameter configuration, `main.py` creates an experiment directory containing subject-level predictions and detailed evaluation results.

Example:

```text
results/
└── casme2_3cls_lr5e-05_wd0.0001_bs128_ep500_seed42/
    ├── predictions/
    │   ├── 1_preds.pt
    │   ├── 2_preds.pt
    │   └── ...
    │
    ├── weights/
    │
    └── detailed_casme2_3cls_....csv
```

The detailed CSV contains:

```text
Subject
ACC
UF1
UAR
Confusion Matrix
```

A global grid-search summary is also generated:

```text
<dataset>_<num_classes>cls_grid_summary_<timestamp>.csv
```

with:

```text
Experiment_ID
LR
WD
BS
EP
Seed
Overall_UF1
Overall_UAR
Overall_ACC
Time
Detail_Path
```

The overall metrics are calculated after concatenating predictions from all LOSO test subjects.

---

## 🔁 Reproducibility

A random seed can be specified using:

```bash
--seed 42
```

The training pipeline controls random states for:

* Python
* NumPy
* PyTorch
* CUDA
* DataLoader sampling

and enables deterministic CuDNN behavior.

---

## ⚠️ Notes

1. **Dataset paths must be modified before running the code.**
   Several paths in `main.py`, `train.py`, and `pre_train.py` are currently configured for the original experimental environment.

2. **Optical flow must be generated in advance.**
   The current repository reads pre-computed onset-to-apex optical-flow images rather than calculating optical flow online.

3. **Multi-dataset pre-training can require substantial storage.**
   The current region perturbation strategy constructs approximately `N²` target-source combinations for `N` successfully detected Apex images.

4. **MTCNN is used for facial landmark localization.**
   If landmark detection fails during downstream optical-flow preprocessing, predefined fallback landmarks are used.

5. **Current supervised checkpoint behavior.**
   `train.py` saves the prediction tensors corresponding to the best validation UF1. The line for saving the supervised model weights is currently commented out and can be enabled if model checkpoints are required.

---

## 📜 Citation

If you find R2-MER useful in your research, please consider citing our work:

```bibtex
@article{r2mer,
  title   = {R2-MER: Micro-Expression Recognition via Region-Aware Reconstruction and Modality-Aware Fusion},
  author  = {Anonymous},
  journal = {Neurocomputing},
  year    = {2026}
}
```

The citation information will be updated after publication.

---

## Acknowledgements

This implementation builds upon PyTorch and uses HTNet as the dynamic optical-flow feature extractor. We thank the authors of the related open-source projects and the creators of the CASME II, SAMM, and SMIC datasets for supporting research in micro-expression analysis.

# R2-MER: Micro-Expression Recognition via Region-Aware Reconstruction and Modality-Aware Fusion

Official PyTorch implementation of **R2-MER**, a two-stage dual-stream framework for Micro-Expression Recognition (MER).

R2-MER first learns region-aware RGB representations through a facial key-region perturbation and reconstruction proxy task, and then integrates static RGB features with dynamic optical-flow features for downstream recognition.

##  Highlights

* **Facial Key-Region Perturbation Pre-training:** swaps four facial regions around the eyes and mouth corners between Apex images and reconstructs the original facial appearance without emotion labels.
* **HKPFE:** preserves fine-grained facial information using parallel high- and low-resolution feature extraction.
* **IRSA:** models interactions between original and perturbed facial representations through cross-region attention.
* **MAFM:** integrates static RGB features with dynamic optical-flow features extracted by HTNet.
* **Multi-Dataset Pre-training:** supports combining multiple datasets into a unified Apex pool for cross-dataset region perturbation.
* **LOSO Evaluation:** automatically reports ACC, UF1, UAR, and confusion matrices.

##  Installation

```bash
git clone https://github.com/Hyposelenia-Kuutar/R2-MER.git
cd R2-MER

pip install -r requirements.txt
pip install pytorch-msssim mediapipe tqdm
```

##  Supported Datasets

The current implementation supports:

* CASME II
* SAMM
* SMIC

RGB frames, onset/apex annotations, class labels, and pre-computed onset-to-apex optical-flow images are required.

## 🚀 Usage

### 1. Region-Recovery Pre-training

Multiple datasets can be combined into a unified pre-training pool:

```bash
python pre_train.py \
    --datasets casme2 smic samm \
    --orig_data_roots /path/to/casme2 /path/to/smic /path/to/samm \
    --csv_paths /path/to/casme2.csv /path/to/smic.csv /path/to/samm.csv \
    --batch_size 128 \
    --epochs 100 \
    --save_path /path/to/pretrain.pth
```

The order of `--datasets`, `--orig_data_roots`, and `--csv_paths` must be consistent.

The default reconstruction objective is:

```text
L_pre = 0.6 × L1 + 0.4 × (1 - SSIM)
```

with AdamW, cosine learning-rate scheduling, and gradient clipping.

### 2. Supervised LOSO Evaluation

Configure dataset paths, optical-flow paths, pretrained weights, GPU, and hyperparameters in `main.py`, then run:

```bash
python main.py
```

The downstream framework uses:

```text
Onset RGB ─┐
           ├─> HKPFE + IRSA ──────────┐
Apex RGB ──┘                           │
                                       ├─> MAFM ─> Classifier
Regional Optical Flow ─> HTNet ───────┘
```

During supervised training, the pretrained RGB branch remains frozen, while HTNet is initially frozen and later unfrozen for joint optimization with MAFM.

## 📊 Evaluation

The LOSO pipeline reports:

* Accuracy (ACC)
* Unweighted F1-score (UF1)
* Unweighted Average Recall (UAR)
* Confusion Matrix

`main.py` automatically performs LOSO evaluation for all subjects and saves subject-level and overall results as CSV files.

##  Repository Structure

```text
R2-MER/
├── main.py           # Full LOSO experiment launcher
├── train.py          # Single-fold training
├── dataset.py        # Dataset loaders
├── Model.py          # HTNet, MAFM, and final model
├── RGB_model.py      # HKPFE, IRSA, and reconstruction model
├── generate_mix.py   # Region perturbation generation
├── pre_train.py      # Reconstruction pre-training
└── requirements.txt
```

##  Notes

* Dataset and result paths should be modified according to your environment.
* Optical-flow images should be generated in advance.
* Multi-dataset pre-training generates approximately `N²` target-source combinations for `N` valid Apex images and may require substantial storage.
* The datasets are not included in this repository.


# R2-MER: Micro-Expression Recognition via Region-Aware Reconstruction and Modality-Aware Fusion

This repository contains the official PyTorch implementation of **R2-MER**, a two-stage, dual-stream framework for Micro-Expression Recognition (MER)[cite: 16]. By introducing a novel self-supervised pre-training proxy task and a dual-stream multi-modal fusion strategy, R2-MER effectively overcomes data scarcity and captures highly subtle facial muscle synergies[cite: 16].

## ✨ Key Features
*   **Facial Key-Region Perturbation Pre-training:** A self-supervised proxy task that swaps four key facial regions (eyes and mouth corners) to force the model to reconstruct perturbed regions, capturing fine-grained structural representations without emotion labels[cite: 16].
*   **Dual-Branch Static Feature Extraction:** Utilizes the Holistic Key-Primed Feature Extractor (HKPFE) and Inter-Region Synergy Attention (IRSA) to explicitly model spatial correlations and multi-scale contextual information in the RGB stream[cite: 16].
*   **Modality-Aware Fusion Module (MAFM):** Efficiently aligns and integrates static appearance features (RGB) with dynamic motion features (Optical Flow from HTNet) to achieve cross-modal complementarity[cite: 16].
*   **Robust LOSO Evaluation:** Built-in pipeline for rigorous Leave-One-Subject-Out cross-validation, automatically tracking metrics like Unweighted F1-score (UF1) and Unweighted Average Recall (UAR)[cite: 16].

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/R2-MER.git](https://github.com/yourusername/R2-MER.git)
   cd R2-MER

# KrivaCare: Multimodal Ovarian Tumor Diagnosis and Explainability Framework

KrivaCare is an advanced diagnostic machine learning pipeline designed for early detection and classification of ovarian tumors. The system uses an uncertainty-aware late fusion architecture to combine data from clinical tabular blood biomarkers and ultrasound image modalities, while delivering local and global model interpretability through Grad-CAM and SHAP integrations.

# System Architecture

1. Tabular Expert: A multi-layer perceptron with batch normalization, dropout, and residual mechanics trained on clinical blood work and patient metrics.

2. Image Expert: A fine-tuned ResNet18 convolutional neural network configured to process ultrasound images and extract deep-seated visual patterns.

3. Uncertainty-Aware Fusion: Dynamically weighs the probabilistic outputs of both expert networks using Shannon entropy to gauge model confidence prior to final classification.

4. Explainability Suite: Generates localized Grad-CAM visual heatmaps for ultrasound scans and localized SHAP contribution bar plots for tabular patient biomarkers.

# Datasets

This project utilizes publicly available benchmark datasets for training and validation:

-    [MMOTU Ovarian Ultrasound Images Dataset](https://www.kaggle.com/datasets/orvile/mmotu-ovarian-ultrasound-images-dataset)

-    [Predict Ovarian Cancer Tabular Dataset](https://www.kaggle.com/datasets/saurabhshahane/predict-ovarian-cancer/code)

## Original References & Literature

1. [Tabular Source Study: Mi, Q., Jiang, J., Znati, T., Fan, Z., Li, J., Xu, B., Chen, L., Zheng, Xiao., & Lu, M. (2020). Data for: Using Machine Learning to Predict Ovarian Cancer. Mendeley Data, V11. DOI: 10.17632/th7fztbrv9.11](https://data.mendeley.com/datasets/th7fztbrv9/11)

2. [Image Source Study: Multi-Modality Ovarian Tumor Ultrasound (MMOTU) dataset collected by the Department of Gynecology and Obstetrics, Beijing Shijitan Hospital, Capital Medical University. Figshare Repository: DOI: 10.6084/m9.figshare.25058690.](https://figshare.com/articles/dataset/_zip/25058690?file=44222642)

# Project Directory Structure

```
KrivaCare/
│
├── config.py             # Global configurations, device settings, and paths
├── data/                 # Raw and processed datasets
├── preprocessing.py      # Cleans tabular records and maps image metadata
├── datasets.py           # Custom PyTorch Dataset classes for data loaders
├── models.py             # TabularExpert and ImageExpert neural network architectures
├── fusion.py             # Uncertainty-aware entropy-based late fusion logic
├── explainability.py     # Grad-CAM and SHAP wrapper functions
├── outputs/              # Model outputs, figures, and results
├── train.py              # Training script with optimization and validation loops
├── inference.py          # End-to-end pipeline execution and visual artifact generation
└── requirements.txt      # Project dependencies
```

# Installation

Clone the repository and install the required dependencies:

```Bash
git clone https://github.com/your-username/KrivaCare.git
cd KrivaCare
pip install -r requirements.txt
```

## Run Inference and Generate Explanations:

Execute the inference pipeline to test random patient records, compute fusion confidence, and output Grad-CAM and SHAP visual explanations to outputs/figures/:

```Bash
python3 inference.py
```

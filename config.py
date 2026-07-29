import os
import torch

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "image")      # Contains preprocessed MMOTU 2D images
TABULAR_DIR = os.path.join(DATA_DIR, "tabular")  # Contains preprocessed Zhengzhou CSVs
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Hardware Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Model Hyperparameters
NUM_CLASSES = 3  # Risk Tier 0, 1, 2
TABULAR_INPUT_DIM = 50 # Adjust based on the exact number of features in your preprocessed CSV
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
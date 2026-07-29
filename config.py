import os
import torch

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "image")      # Contains MMOTU OTU_2d images
TABULAR_DIR = os.path.join(DATA_DIR, "tabular")  # Contains raw Excel files
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Hardware Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Model Hyperparameters
NUM_CLASSES = 2  # Binary: 0 = Malignant, 1 = Benign
TABULAR_INPUT_DIM = 50 
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 40

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
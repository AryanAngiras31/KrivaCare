import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import joblib
import matplotlib.pyplot as plt
import shap

# Import from your local project modules
import config
from models import ImageExpert, TabularExpert
from fusion import uncertainty_aware_fusion
from explainability import generate_gradcam, generate_shap_values

class ClinicalInferencePipeline:
    def __init__(self):
        self.device = config.DEVICE
        print(f"Initializing Inference Pipeline on {self.device}...")

        # 1. Load Preprocessing Artifacts
        self.scaler = joblib.load(os.path.join(config.ARTIFACTS_DIR, 'tabular_scaler.pkl'))
        self.medians = pd.read_pickle(os.path.join(config.ARTIFACTS_DIR, 'tabular_medians.pkl'))
        self.feature_cols = self.medians.index.tolist()

        # 2. Load Trained Tabular Expert
        self.tabular_model = TabularExpert(input_dim=len(self.feature_cols), num_classes=config.NUM_CLASSES).to(self.device)
        self.tabular_model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'tabular_expert.pth'), map_location=self.device))
        self.tabular_model.eval()

        # 3. Load Trained Image Expert
        self.image_model = ImageExpert(num_classes=config.NUM_CLASSES).to(self.device)
        self.image_model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'image_expert.pth'), map_location=self.device))
        self.image_model.eval()

        # 4. Standard Image Transforms for PyTorch
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 5. Load Background Data for SHAP Explainer
        # SHAP requires a background distribution to calculate baseline expectations. 
        # We load a small sample (100 rows) from the preprocessed training set.
        train_csv_path = os.path.join(config.TABULAR_DIR, 'cleaned_train.csv')
        if os.path.exists(train_csv_path):
            bg_df = pd.read_csv(train_csv_path).drop(columns=['RiskTier'], errors='ignore').head(100)
            self.bg_tensor = torch.tensor(bg_df.values, dtype=torch.float32).to(self.device)
        else:
            print("Warning: cleaned_train.csv not found. SHAP values may fail if background data is missing.")
            self.bg_tensor = None

    def preprocess_patient_vitals(self, patient_dict):
        """Cleans, imputes, and scales a single patient's raw clinical dictionary."""
        df = pd.DataFrame([patient_dict])
        
        # Ensure all required features exist; impute missing ones with training medians
        for col in self.feature_cols:
            if col not in df.columns or pd.isna(df[col].iloc[0]):
                df[col] = self.medians[col]
                
        # Enforce correct feature order
        ordered_features = df[self.feature_cols].values
        
        # Scale based on the training distribution
        scaled_features = self.scaler.transform(ordered_features)
        return torch.tensor(scaled_features, dtype=torch.float32).to(self.device), df[self.feature_cols]

    def preprocess_image(self, image_path):
        """Loads and processes the ultrasound image for both the model and Grad-CAM."""
        img = Image.open(image_path).convert('RGB')
        
        # 1. Tensor for the model
        img_tensor = self.img_transform(img).unsqueeze(0).to(self.device)
        
        # 2. Normalized Numpy Array for Grad-CAM Overlay ([0, 1] range)
        img_resized = img.resize((224, 224))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        
        return img_tensor, img_np

    def predict_and_explain(self, patient_dict, image_path):
        print("\n--- Running Inference & Fusion ---")
        
        # 1. Preprocess Inputs
        tab_tensor, feature_df = self.preprocess_patient_vitals(patient_dict)
        img_tensor, img_np = self.preprocess_image(image_path)

        # 2. Get Model Predictions
        with torch.no_grad():
            tab_logits = self.tabular_model(tab_tensor)
            img_logits = self.image_model(img_tensor)
            
            # Uncertainty-Aware Fusion
            fused_probs, final_pred = uncertainty_aware_fusion(img_logits, tab_logits)
            
            # Extract probabilities
            prob_malignant = fused_probs[0][0].item() * 100
            prob_benign = fused_probs[0][1].item() * 100
            diagnosis = "Benign" if final_pred.item() == 1 else "Malignant"

        print(f"Final Diagnosis: {diagnosis}")
        print(f"Confidence: Malignant ({prob_malignant:.2f}%) | Benign ({prob_benign:.2f}%)")

        # 3. Generate Explainability
        print("\n--- Generating AI Explanations ---")
        
        # Grad-CAM (Requires gradients, so it runs outside no_grad)
        cam_visualization = generate_gradcam(self.image_model, img_tensor, img_np)
        
        # SHAP (DeepExplainer)
        shap_values = None
        if self.bg_tensor is not None:
            shap_values = generate_shap_values(self.tabular_model, self.bg_tensor, tab_tensor)

        return diagnosis, fused_probs, cam_visualization, shap_values, feature_df

    def visualize_results(self, image_path, cam_visualization, shap_values, feature_df):
        """Helper to plot and save the results."""
        # Ensure the figures directory exists
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        
        # 1. Show Original Image vs Grad-CAM
        original_img = Image.open(image_path).convert('RGB').resize((224, 224))
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_img)
        axes[0].set_title("Original Ultrasound")
        axes[0].axis('off')
        
        axes[1].imshow(cam_visualization)
        axes[1].set_title("ResNet18 Grad-CAM Heatmap")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save Grad-CAM result
        cam_save_path = os.path.join(config.FIGURES_DIR, 'gradcam_result.png')
        plt.savefig(cam_save_path, bbox_inches='tight')
        print(f"Saved Grad-CAM visualization to {cam_save_path}")
        plt.close() # Close figure to free memory

        # 2. Show SHAP Force Plot for Tabular Features
        if shap_values is not None:
            print("\nSHAP Feature Importances (Tabular Model):")
            shap.initjs()
            
            # Convert to numpy for plotting
            shap_numpy = shap_values[0] if isinstance(shap_values, list) else shap_values
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_numpy, 
                feature_df, 
                feature_names=self.feature_cols, 
                plot_type="bar",
                show=False # Prevent automatic display so we can save it
            )
            
            # Save SHAP result
            shap_save_path = os.path.join(config.FIGURES_DIR, 'shap_result.png')
            plt.savefig(shap_save_path, bbox_inches='tight')
            print(f"Saved SHAP visualization to {shap_save_path}")
            plt.close()

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    import random
    
    pipeline = ClinicalInferencePipeline()
    
    print("\n--- Picking Random Patient Data ---")
    # 1. Pick a random patient from the raw test set
    test_tabular_path = os.path.join(config.TABULAR_DIR, 'tabular_test.xlsx')
    if os.path.exists(test_tabular_path):
        test_df = pd.read_excel(test_tabular_path)
        random_idx = random.randint(0, len(test_df) - 1)
        patient_row = test_df.iloc[random_idx].to_dict()
        
        # Remove target and ID columns from the dictionary input
        if 'TYPE' in patient_row:
            actual_label = patient_row.pop('TYPE')
            print(f"Patient {random_idx} Actual Tabular Label: {'Benign' if actual_label == 1 else 'Malignant'}")
        if 'SUBJECT_ID' in patient_row:
            patient_row.pop('SUBJECT_ID')
    else:
        print(f"{test_tabular_path} not found. Using hardcoded fallback.")
        patient_row = {'Age': 58, 'CA125': 2800.5, 'Menopause': 1, 'HE4': 850.0}

    # 2. Pick a random image from the validation set
    val_image_csv = os.path.join(config.IMAGE_DIR, 'mmotu_val.csv')
    if os.path.exists(val_image_csv):
        val_img_df = pd.read_csv(val_image_csv)
        random_img_idx = random.randint(0, len(val_img_df) - 1)
        sample_image_path = val_img_df.iloc[random_img_idx]['image_path']
        actual_img_label = val_img_df.iloc[random_img_idx]['label']
        print(f"Random Image Actual Label: {'Benign' if actual_img_label == 1 else 'Malignant'}")
    else:
        print(f"{val_image_csv} not found. Using fallback.")
        sample_image_path = os.path.join(config.IMAGE_DIR, 'images', '1.PNG') 

    print(f"\nRunning inference on Image: {os.path.basename(sample_image_path)}")
    
    if os.path.exists(sample_image_path):
        diagnosis, probs, cam, shap_vals, df_feats = pipeline.predict_and_explain(
            patient_row, 
            sample_image_path
        )
        
        # Render and save the visualizations
        pipeline.visualize_results(sample_image_path, cam, shap_vals, df_feats)
    else:
        print(f"{sample_image_path} not found.")
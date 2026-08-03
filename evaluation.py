import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Import from your local project modules
import config
from inference import ClinicalInferencePipeline
from fusion import uncertainty_aware_fusion

def set_seed(seed=42):
    """Ensures reproducibility for the random pairing."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_simulated_pairs(tabular_path, image_path):
    print("--- Generating Simulated Paired Test Set ---")
    
    # Load raw datasets
    tab_df = pd.read_excel(tabular_path)
    img_df = pd.read_csv(image_path)
    
    # Separate by Class (Assuming 1 = Benign, 0 = Malignant based on your inference logic)
    tab_benign = tab_df[tab_df['TYPE'] == 1]
    tab_malignant = tab_df[tab_df['TYPE'] == 0]
    
    img_benign = img_df[img_df['label'] == 1]
    img_malignant = img_df[img_df['label'] == 0]
    
    # Determine the maximum possible pairs (bottlenecked by the smaller dataset)
    num_benign = min(len(tab_benign), len(img_benign))
    num_malignant = min(len(tab_malignant), len(img_malignant))
    
    print(f"Synthesizing {num_benign} Benign pairs and {num_malignant} Malignant pairs...")
    
    # Shuffle the datasets to ensure random pairing
    tab_benign = tab_benign.sample(frac=1, random_state=42).reset_index(drop=True)
    img_benign = img_benign.sample(frac=1, random_state=42).reset_index(drop=True)
    
    tab_malignant = tab_malignant.sample(frac=1, random_state=42).reset_index(drop=True)
    img_malignant = img_malignant.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create the pairs
    paired_data = []
    
    # Process Benign
    for i in range(num_benign):
        tab_row = tab_benign.iloc[i].to_dict()
        tab_row.pop('TYPE', None)
        tab_row.pop('SUBJECT_ID', None)
        
        paired_data.append({
            'patient_id': f"sim_benign_{i}",
            'true_label': 1,
            'image_path': img_benign.iloc[i]['image_path'],
            'tabular_dict': tab_row
        })
        
    # Process Malignant
    for i in range(num_malignant):
        tab_row = tab_malignant.iloc[i].to_dict()
        tab_row.pop('TYPE', None)
        tab_row.pop('SUBJECT_ID', None)
        
        paired_data.append({
            'patient_id': f"sim_malignant_{i}",
            'true_label': 0,
            'image_path': img_malignant.iloc[i]['image_path'],
            'tabular_dict': tab_row
        })
        
    # Shuffle the final paired dataset
    random.shuffle(paired_data)
    return paired_data

def evaluate_models():
    set_seed(42)
    
    # Ensure outputs directories exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    results_txt_path = os.path.join(config.RESULTS_DIR, "evaluation_results.txt")
    results_csv_path = os.path.join(config.RESULTS_DIR, "evaluation_results.csv")
    
    # 1. Initialize Pipeline
    pipeline = ClinicalInferencePipeline()
    
    # 2. Generate Pairs
    test_tabular_path = os.path.join(config.TABULAR_DIR, 'tabular_test.xlsx')
    val_image_csv = os.path.join(config.IMAGE_DIR, 'mmotu_val.csv')
    
    if not os.path.exists(test_tabular_path) or not os.path.exists(val_image_csv):
        print("Error: Test datasets not found. Please check paths.")
        return
        
    paired_dataset = generate_simulated_pairs(test_tabular_path, val_image_csv)
    
    # 3. Tracking Arrays
    y_true = []
    y_pred_image = []
    y_pred_tabular = []
    y_pred_fusion = []
    
    print("\n--- Running Evaluation on Simulated Test Set ---")
    
    pipeline.image_model.eval()
    pipeline.tabular_model.eval()
    
    with torch.no_grad():
        for item in tqdm(paired_dataset, desc="Evaluating Patients"):
            # Load Data
            true_label = item['true_label']
            img_path = item['image_path']
            tab_dict = item['tabular_dict']
            
            # Preprocess using your existing logic
            try:
                tab_tensor, _ = pipeline.preprocess_patient_vitals(tab_dict)
                img_tensor, _ = pipeline.preprocess_image(img_path)
            except Exception as e:
                continue
                
            # Get Individual Logits
            img_logits = pipeline.image_model(img_tensor)
            tab_logits = pipeline.tabular_model(tab_tensor)
            
            # Get Predictions
            img_pred = torch.argmax(img_logits, dim=1).item()
            tab_pred = torch.argmax(tab_logits, dim=1).item()
            
            # Get Fusion Prediction
            _, fusion_pred = uncertainty_aware_fusion(img_logits, tab_logits)
            fusion_pred = fusion_pred.item()
            
            # Store
            y_true.append(true_label)
            y_pred_image.append(img_pred)
            y_pred_tabular.append(tab_pred)
            y_pred_fusion.append(fusion_pred)

    # 4. Generate Reports and Dataframes
    target_names = ['Malignant (0)', 'Benign (1)']
    
    # Calculate pure accuracy
    acc_img = accuracy_score(y_true, y_pred_image)
    acc_tab = accuracy_score(y_true, y_pred_tabular)
    acc_fus = accuracy_score(y_true, y_pred_fusion)
    
    # Generate string reports for the text file
    report_img_str = classification_report(y_true, y_pred_image, target_names=target_names, digits=4)
    report_tab_str = classification_report(y_true, y_pred_tabular, target_names=target_names, digits=4)
    report_fus_str = classification_report(y_true, y_pred_fusion, target_names=target_names, digits=4)
    
    # Generate dictionary reports for parsing into Pandas
    report_img_dict = classification_report(y_true, y_pred_image, target_names=target_names, output_dict=True)
    report_tab_dict = classification_report(y_true, y_pred_tabular, target_names=target_names, output_dict=True)
    report_fus_dict = classification_report(y_true, y_pred_fusion, target_names=target_names, output_dict=True)
    
    # Create the DataFrame
    results_data = [
        {
            "Model Architecture": "Image-Only Expert (ResNet18)",
            "Accuracy": acc_img,
            "Precision (Macro)": report_img_dict['macro avg']['precision'],
            "Recall (Macro)": report_img_dict['macro avg']['recall'],
            "F1-Score (Macro)": report_img_dict['macro avg']['f1-score']
        },
        {
            "Model Architecture": "Tabular-Only Expert (MLP)",
            "Accuracy": acc_tab,
            "Precision (Macro)": report_tab_dict['macro avg']['precision'],
            "Recall (Macro)": report_tab_dict['macro avg']['recall'],
            "F1-Score (Macro)": report_tab_dict['macro avg']['f1-score']
        },
        {
            "Model Architecture": "Uncertainty-Aware Fusion",
            "Accuracy": acc_fus,
            "Precision (Macro)": report_fus_dict['macro avg']['precision'],
            "Recall (Macro)": report_fus_dict['macro avg']['recall'],
            "F1-Score (Macro)": report_fus_dict['macro avg']['f1-score']
        }
    ]
    
    df_results = pd.DataFrame(results_data)
    
    # Round the metrics for cleaner presentation in the CSV
    df_results = df_results.round(4)
    
    # Save the text report
    output_text = (
        "==========================================\n"
        "MULTIMODAL EVALUATION RESULTS\n"
        "==========================================\n\n"
        "1. IMAGE-ONLY EXPERT PERFORMANCE\n"
        f"{report_img_str}\n"
        "==========================================\n"
        "2. TABULAR-ONLY EXPERT PERFORMANCE\n"
        f"{report_tab_str}\n"
        "==========================================\n"
        "3. UNCERTAINTY-AWARE FUSION PERFORMANCE\n"
        f"{report_fus_str}\n"
        "==========================================\n"
    )
    
    print(f"\n{output_text}")
    
    with open(results_txt_path, "w") as f:
        f.write(output_text)
        
    # Save the Pandas DataFrame to CSV
    df_results.to_csv(results_csv_path, index=False)
    
    print(f"Full text evaluation report saved to: {results_txt_path}")
    print(f"Results DataFrame saved to: {results_csv_path}")

if __name__ == "__main__":
    evaluate_models()
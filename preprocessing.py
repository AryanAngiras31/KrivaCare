import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from config import TABULAR_DIR, ARTIFACTS_DIR, IMAGE_DIR

# MMOTU Raw Label (0-7) to Binary Mapping (0: Malignant, 1: Benign)
MMOTU_TO_BINARY = {
    0: 1,  # Endometrioma -> Benign
    1: 1,  # Serous cystadenoma -> Benign
    2: 1,  # Teratoma -> Benign
    3: 1,  # Theca cell tumor -> Benign
    4: 1,  # Simple cyst -> Benign
    5: 1,  # Normal ovary -> Benign
    6: 1,  # Mucinous cystadenoma -> Benign
    7: 0,  # High-grade serous carcinoma -> Malignant
}

def clean_tabular_data(df):
    if 'SUBJECT_ID' in df.columns:
        df = df.drop(columns=['SUBJECT_ID'])
        
    cols_to_clean = ['AFP', 'CA125', 'CA19-9']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('>', '', regex=False)
            df[col] = df[col].astype(str).str.replace('\t', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')            
    return df

def run_tabular_preprocessing():
    print("--- 1. Preprocessing Tabular Data ---")
    train_path = os.path.join(TABULAR_DIR, 'tabular_train.xlsx')
    test_path = os.path.join(TABULAR_DIR, 'tabular_test.xlsx')
    
    # 1. Load both datasets
    df1 = pd.read_excel(train_path, sheet_name=0)
    df2 = pd.read_excel(test_path, sheet_name=0)
    
    # 2. Combine them into a single dataset
    df_all = pd.concat([df1, df2], ignore_index=True)
    
    # 3. Clean string artifacts and drop IDs
    df_all = clean_tabular_data(df_all)
    
    # Separate features and target
    X = df_all.drop(columns=['TYPE'])
    y = df_all['TYPE']
    feature_cols = X.columns.tolist()
    
    # 4. Perform a Stratified 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # 5. Calculate medians STRICTLY on the training set to prevent data leakage
    medians = X_train.median()
    
    # 6. Impute missing values
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    
    # 7. Feature Scaling (Fit on train, transform on both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    
    # 8. Rebuild DataFrames for saving
    train_processed = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_processed['RiskTier'] = y_train.values
    
    test_processed = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_processed['RiskTier'] = y_test.values
    
    # Save cleaned CSV files into TABULAR_DIR
    train_processed.to_csv(os.path.join(TABULAR_DIR, 'cleaned_train.csv'), index=False)
    test_processed.to_csv(os.path.join(TABULAR_DIR, 'cleaned_test.csv'), index=False)
    
    # Save artifacts cleanly
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'tabular_scaler.pkl'))
    medians.to_pickle(os.path.join(ARTIFACTS_DIR, 'tabular_medians.pkl'))
    
    print(f"Tabular data preprocessed & saved successfully.")
    print(f"New Train Size: {len(train_processed)} | New Test Size: {len(test_processed)}")

def run_image_metadata_preprocessing():
    print("--- 2. Preprocessing MMOTU Image Metadata ---")
    
    def process_split_file(txt_filename, output_csv_filename):
        txt_path = os.path.join(IMAGE_DIR, txt_filename)
        if not os.path.exists(txt_path):
            print(f"Warning: Could not find {txt_path}")
            return
            
        records = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name, raw_label = parts[0], int(parts[1])
                    img_full_path = os.path.join(IMAGE_DIR, "images", img_name)
                    binary_label = MMOTU_TO_BINARY.get(raw_label, 1)
                    
                    records.append({
                        "image_path": img_full_path,
                        "raw_label": raw_label,
                        "label": binary_label
                    })
                    
        df = pd.DataFrame(records)
        save_path = os.path.join(IMAGE_DIR, output_csv_filename)
        df.to_csv(save_path, index=False)
        print(f"Created {output_csv_filename} ({len(df)} images mapped)")

    process_split_file("train_cls.txt", "mmotu_train.csv")
    process_split_file("val_cls.txt", "mmotu_val.csv")

if __name__ == "__main__":
    run_tabular_preprocessing()
    run_image_metadata_preprocessing()
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from  config import OUTPUT_DIR

def clean_tabular_data(df, is_train=True):
    # 1. Drop unused or identifying columns
    if 'SUBJECT_ID' in df.columns:
        df = df.drop(columns=['SUBJECT_ID'])
        
    # 2. Fix the string artifacts in the specific columns
    cols_to_clean = ['AFP', 'CA125', 'CA19-9']
    for col in cols_to_clean:
        if col in df.columns:
            # Strip trailing tabs and greater-than signs
            df[col] = df[col].astype(str).str.replace('>', '', regex=False)
            df[col] = df[col].astype(str).str.replace('\t', '', regex=False)
            # Force to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def run_preprocessing(train_path, test_path, output_dir):
    print("--- Starting Tabular Preprocessing ---")
    
    # Load raw excel sheets
    train_df = pd.read_excel(train_path, sheet_name=0)
    test_df = pd.read_excel(test_path, sheet_name=0)
    
    # Clean string artifacts
    train_df = clean_tabular_data(train_df)
    test_df = clean_tabular_data(test_df)
    
    # 3. Handle Missing Values using Train medians
    # Important: We must calculate medians strictly on the training set to prevent data leakage.
    feature_cols = [c for c in train_df.columns if c != 'TYPE']
    medians = train_df[feature_cols].median()
    
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    test_df[feature_cols] = test_df[feature_cols].fillna(medians)
    
    # Extract labels and features
    y_train = train_df['TYPE'].values
    X_train_raw = train_df[feature_cols].values
    
    y_test = test_df['TYPE'].values
    X_test_raw = test_df[feature_cols].values
    
    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Rebuild DataFrames for saving
    train_processed = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_processed['RiskTier'] = y_train  # Rename label column for consistency
    
    test_processed = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_processed['RiskTier'] = y_test
    
    # Create outputs dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the cleaned datasets
    train_processed.to_csv(os.path.join(output_dir, 'cleaned_train.csv'), index=False)
    test_processed.to_csv(os.path.join(output_dir, 'cleaned_test.csv'), index=False)
    
    # Save the scaler and medians for future inference
    joblib.dump(scaler, os.path.join(output_dir, 'tabular_scaler.pkl'))
    medians.to_pickle(os.path.join(output_dir, 'tabular_medians.pkl'))
    
    print(f"✅ Processed Train Rows: {len(train_processed)}")
    print(f"✅ Processed Test Rows:  {len(test_processed)}")
    print(f"✅ Preprocessing artifacts saved to {output_dir}")

if __name__ == "__main__":
    run_preprocessing(
        train_path='tabular_train.xlsx',
        test_path='tabular_test.xlsx',
        output_dir='data/tabular'
    )
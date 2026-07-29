import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_clinical_data(filepath):
    df = pd.read_excel(filepath)
    
    # 1. Drop the ID column
    if 'SUBJECT_ID' in df.columns:
        df = df.drop(columns=['SUBJECT_ID'])
        
    # 2. Clean string artifacts in specific columns
    for col in ['AFP', 'CA125', 'CA19-9']:
        if col in df.columns:
            # Remove > and \t characters
            df[col] = df[col].astype(str).str.replace('>', '', regex=False)
            df[col] = df[col].astype(str).str.replace('\t', '', regex=False)
            # Convert back to numeric, coercing any remaining errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 3. Impute missing values with the column median
    df = df.fillna(df.median())
    
    return df

# Load and clean
train_df = preprocess_clinical_data('tabular_train.xlsx')
test_df = preprocess_clinical_data('tabular_test.xlsx')

# 4. Scale the features (excluding the target 'TYPE')
scaler = StandardScaler()
X_train = train_df.drop(columns=['TYPE'])
y_train = train_df['TYPE']

X_test = test_df.drop(columns=['TYPE'])
y_test = test_df['TYPE']

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
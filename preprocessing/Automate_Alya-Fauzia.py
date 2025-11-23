import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump

def load_data(path):
    """Load dataset dari file CSV."""
    return pd.read_csv(path)

def split_numeric_categorical(df):
    """Pisah kolom numerik dan kategorikal."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    return numeric_cols, categorical_cols

def handle_outliers(df, numeric_cols):
    """Handling outlier menggunakan metode IQR."""
    df_clean = df.copy()

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    return df_clean

def encode_categorical(df, categorical_cols):
    """Label encoding untuk kolom kategorikal."""
    df_encoded = df.copy()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    return df_encoded, encoders

def standard_scaler(df, numeric_cols):
    """Scaler untuk fitur numerik."""
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled, scaler

def automate_preprocessing(path, target_col, output_dir, drop_data):
    """Automasi preprocessing lengkap."""

    # === 0. Buat folder jika belum ada ===
    os.makedirs(output_dir, exist_ok=True)

    # === 1. Load data ===
    df = load_data(path)
    df = df.drop(columns=drop_data)

    # === 2. Split numeric & categorical ===
    numeric_cols, categorical_cols = split_numeric_categorical(df.drop(columns=[target_col]))

    # === 3. Handling outliers ===
    df = handle_outliers(df, numeric_cols)

    # === 4. Encode kategorikal ===
    df_encoded, encoders = encode_categorical(df, categorical_cols)

    # === 5. Train-test split ===
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === 6. Scaling numerik ===
    X_train_scaled, scaler = standard_scaler(X_train, numeric_cols)
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # === 7. Simpan train & test CSV ===
    df_train = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    df_test = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)

    df_train.to_csv(f"{output_dir}/train_data.csv", index=False)
    df_test.to_csv(f"{output_dir}/test_data.csv", index=False)

    # === 8. Simpan encoder & scaler ===
    dump({"encoders": encoders, "scaler": scaler, 
          "numeric_cols": numeric_cols,
          "categorical_cols": categorical_cols},
          f"{output_dir}/preprocessor.joblib")

    print("Preprocessing otomatis selesai!")

    return X_train_scaled, X_test_scaled, y_train, y_test, encoders


if __name__ == "__main__":
    data = "LoanEligibilityPrediction.csv"
    target = "Loan_Status"
    output_dir = 'dataset_preprocessing'
    drop_col = ['Customer_ID', 'Education', 'Property_Area', 'Gender']

    automate_preprocessing(
        path=data, 
        target_col=target,
        output_dir=output_dir,
        drop_data=drop_col
    )

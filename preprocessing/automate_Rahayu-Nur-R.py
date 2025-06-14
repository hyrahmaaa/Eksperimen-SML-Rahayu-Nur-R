
# automate_TelcoChurn_Nama-siswa.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

RAW_DATA_PATH = os.path.join('telco_churn_raw', 'Telco-Customer-Churn.csv')
PROCESSED_DATA_DIR = os.path.join('preprocessing', 'telco_churn_preprocessing')

def load_data(file_path):
    """
    Memuat dataset dari path yang diberikan.
    """
    print(f"Memuat data dari: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data dimuat. Ukuran: {df.shape}")
    return df
    
def handle_total_charges(df):
    """
    Menangani kolom TotalCharges: mengubah ke numerik dan mengisi missing values.
    """
    print("Menangani kolom 'TotalCharges'...")
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_total_charges, inplace=True) # Perhatian: warning inplace tidak masalah saat ini
    
    print(f"NaN di 'TotalCharges' diisi dengan median: {median_total_charges:.2f}")
    print(f"Tipe data 'TotalCharges' setelah konversi: {df['TotalCharges'].dtype}")
    
    return df
    
def encode_binary_features(df):
    """
    Menerapkan Label Encoding pada fitur kategorikal biner.
    """
    print("Menerapkan Label Encoding pada fitur biner...")
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn'] # <-- Sesuaikan jika daftar Anda berbeda
    le = LabelEncoder()
    for col in binary_cols:
        if df[col].dtype == 'object': # Pastikan kolom adalah objek sebelum encoding
            df[col] = le.fit_transform(df[col])
            # Print opsional untuk verifikasi di script
            # print(f"  Kolom '{col}' di-encode. Nilai unik: {df[col].unique()}")
    print("Label Encoding selesai.")
    return df
    
def encode_multi_class_features(df):
    """
    Menerapkan One-Hot Encoding pada fitur kategorikal multi-kelas.
    """
    print("Menerapkan One-Hot Encoding pada fitur multi-kelas...")
    multi_cols = [ 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]

    multi_cols_to_encode = [col for col in multi_cols if col in df.columns and df[col].dtype == 'object']

    df = pd.get_dummies(df, columns=multi_cols_to_encode, drop_first=True) 
    
    print("One-Hot Encoding selesai.")
    return df

def handle_outliers(df):
    """
    Menangani outlier pada kolom numerik menggunakan metode capping IQR.
    """
    print("Menangani outlier pada fitur numerik (capping IQR)...")
    numerical_cols = ['SeniorCitizen','tenure', 'MonthlyCharges', 'TotalCharges']

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        print(f"  Outlier di '{col}' ditangani (Batas: {lower_bound:.2f} - {upper_bound:.2f})")

    print("Penanganan outlier selesai.")
    return df

def split_and_scale_data(df):
    """
    Membagi data menjadi training dan testing, lalu melakukan penskalaan.
    """
    print("Membagi dan menskalakan data...")

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numerical_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    numerical_cols_to_scale = [col for col in numerical_cols_to_scale if col in X_train.columns]

    scaler = StandardScaler()

    X_train[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
    X_test[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])

    print("Split dan penskalaan selesai.")
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    """
    Menyimpan data yang sudah diproses ke file CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    print(f"Data yang diproses disimpan di: {output_dir}")

if __name__ == "__main__":
    print("--- Memulai Otomasi Preprocessing Data Telco Churn ---")

    df = load_data(RAW_DATA_PATH)
    df = handle_total_charges(df.copy())

    df = encode_binary_features(df.copy())
    df = encode_multi_class_features(df.copy())

    df = handle_outliers(df.copy())

    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(df.copy())

    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, PROCESSED_DATA_DIR)

    print("\n--- Otomasi Preprocessing Selesai! ---")

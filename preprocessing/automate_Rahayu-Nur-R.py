# automate_TelcoChurn_Nama-siswa.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = 'Telco-Customer-Churn.csv' 
PROCESSED_DATA_DIR = os.path.join('preprocessing', 'telco_churn_preprocessing')

BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
MULTI_CLASS_COLS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
NUMERIC_COLS_FOR_OUTLIERS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
NUMERIC_COLS_FOR_SCALING = ['tenure', 'MonthlyCharges', 'TotalCharges'] 

def load_data(file_path):
    """
    Memuat dataset Telco Customer Churn dari file CSV yang diberikan.

    Args:
        file_path (str): Path lengkap ke file CSV dataset.

    Returns:
        pd.DataFrame: DataFrame yang dimuat, atau None jika gagal.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset berhasil dimuat dari '{file_path}'. Dimensi awal: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di path '{file_path}'. Pastikan path sudah benar.")
        return None
    except pd.errors.EmptyDataError:
        print(f"ERROR: File '{file_path}' kosong.")
        return None
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan saat memuat data dari '{file_path}': {e}")
        return None

def handle_total_charges(df):
    """
    Menangani kolom 'TotalCharges': Mengonversi ke tipe data numerik
    dan mengisi nilai yang hilang (space, NaN) dengan median.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame dengan kolom 'TotalCharges' yang sudah diproses.
    """
    print("\n--- Memulai Penanganan Kolom 'TotalCharges' ---")
    if 'TotalCharges' not in df.columns:
        print("PERINGATAN: Kolom 'TotalCharges' tidak ditemukan. Lewati penanganan.")
        return df

    df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    median_total_charges = df['TotalCharges'].median()
    initial_nan_count = df['TotalCharges'].isnull().sum()

    if initial_nan_count > 0:
        df['TotalCharges'].fillna(median_total_charges, inplace=True)
        print(f"  {initial_nan_count} nilai hilang di 'TotalCharges' diisi dengan median ({median_total_charges:.2f}).")
    else:
        print("  Tidak ada nilai hilang di 'TotalCharges' yang perlu diisi.")

    print(f"  Tipe data 'TotalCharges' setelah konversi: {df['TotalCharges'].dtype}")
    print("--- Penanganan Kolom 'TotalCharges' Selesai ---")
    return df

def encode_binary_features(df, cols_to_encode):
    """
    Menerapkan Label Encoding pada fitur kategorikal biner yang ditentukan.

    Args:
        df (pd.DataFrame): DataFrame input.
        cols_to_encode (list): Daftar nama kolom biner yang akan di-encode.

    Returns:
        pd.DataFrame: DataFrame dengan kolom biner yang sudah di-encode.
    """
    print("\n--- Memulai Label Encoding Fitur Biner ---")
    le = LabelEncoder()
    encoded_cols = []
    for col in cols_to_encode:
        if col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                df[col] = le.fit_transform(df[col])
                encoded_cols.append(col)
            else:
                print(f"  PERINGATAN: Kolom '{col}' bukan tipe objek/kategorikal. Lewati Label Encoding.")
        else:
            print(f"  PERINGATAN: Kolom '{col}' tidak ditemukan. Lewati Label Encoding.")

    if encoded_cols:
        print(f"  Kolom biner berikut telah di-Label Encode: {encoded_cols}")
    else:
        print("  Tidak ada kolom biner yang berhasil di-Label Encode.")
    print("--- Label Encoding Fitur Biner Selesai ---")
    return df

def encode_multi_class_features(df, cols_to_encode):
    """
    Menerapkan One-Hot Encoding pada fitur kategorikal multi-kelas yang ditentukan.

    Args:
        df (pd.DataFrame): DataFrame input.
        cols_to_encode (list): Daftar nama kolom multi-kelas yang akan di-encode.

    Returns:
        pd.DataFrame: DataFrame dengan kolom multi-kelas yang sudah di-One-Hot Encode.
    """
    print("\n--- Memulai One-Hot Encoding Fitur Multi-Kelas ---")
    multi_cols_found = [
        col for col in cols_to_encode
        if col in df.columns and (df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]))
    ]

    if multi_cols_found:
        df = pd.get_dummies(df, columns=multi_cols_found, drop_first=True)
        print(f"  Kolom multi-kelas berikut telah di-One-Hot Encode (drop_first=True): {multi_cols_found}")
    else:
        print("  Tidak ada kolom multi-kelas yang ditemukan atau yang perlu di-One-Hot Encode.")

    print("--- One-Hot Encoding Fitur Multi-Kelas Selesai ---")
    return df

def handle_outliers_iqr_clip(df, columns_to_treat):
    """
    Menangani outlier pada kolom numerik menggunakan metode IQR clipping.

    Args:
        df (pd.DataFrame): DataFrame input.
        columns_to_treat (list): Daftar nama kolom numerik untuk penanganan outlier.

    Returns:
        pd.DataFrame: DataFrame dengan outlier yang sudah di-clip.
    """
    print("\n--- Memulai Penanganan Outlier (IQR Clipping) ---")
    treated_cols = []
    for col in columns_to_treat:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_before = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            if outliers_before > 0:
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                treated_cols.append(col)
                print(f"  Outlier di '{col}' telah di-clip ke rentang [{lower_bound:.2f}, {upper_bound:.2f}]. {outliers_before} nilai terpengaruh.")
            else:
                print(f"  Tidak ada outlier signifikan di '{col}' yang memerlukan clipping.")
        else:
            print(f"  PERINGATAN: Kolom '{col}' tidak ditemukan atau bukan numerik. Lewati penanganan outlier.")

    if treated_cols:
        print(f"  Outlier telah ditangani untuk kolom: {treated_cols}")
    else:
        print("  Tidak ada kolom yang memerlukan penanganan outlier atau tidak ada kolom numerik yang valid.")
    print("--- Penanganan Outlier Selesai ---")
    return df

def split_and_scale_data(df, target_column='Churn', cols_to_scale=None, id_column='customerID'):
    """
    Membagi data menjadi training dan testing set (X dan y),
    kemudian melakukan penskalaan pada fitur numerik.

    Args:
        df (pd.DataFrame): DataFrame yang sudah diproses.
        target_column (str): Nama kolom target (label).
        cols_to_scale (list): Daftar nama kolom numerik yang akan di-scale.
        id_column (str): Nama kolom ID yang akan dihapus sebelum splitting.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) data yang sudah di-split dan di-scale.
    """
    print("\n--- Memulai Split dan Scaling Data ---")

    if id_column in df.columns:
        print(f"  Menghapus kolom '{id_column}'.")
        df = df.drop(columns=[id_column])
    else:
        print(f"  Kolom '{id_column}' tidak ditemukan, lewati penghapusan.")

    if target_column not in df.columns:
        print(f"ERROR: Kolom target '{target_column}' tidak ditemukan. Tidak dapat melakukan split.")
        return None, None, None, None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Data berhasil dibagi: X_train={X_train.shape}, X_test={X_test.shape}")

    if cols_to_scale:
        numerical_features_to_scale = [col for col in cols_to_scale if col in X_train.columns and pd.api.types.is_numeric_dtype(X_train[col])]
        if numerical_features_to_scale:
            scaler = StandardScaler()
            X_train[numerical_features_to_scale] = scaler.fit_transform(X_train[numerical_features_to_scale])
            X_test[numerical_features_to_scale] = scaler.transform(X_test[numerical_features_to_scale])
            print(f"  Fitur numerik berikut telah di-scale: {numerical_features_to_scale}")
        else:
            print("  Tidak ada fitur numerik yang valid untuk di-scale dari daftar yang diberikan.")
    else:
        print("  Tidak ada kolom yang ditentukan untuk penskalaan.")

    print("--- Split dan Scaling Data Selesai ---")
    return X_train, X_test, y_train, y_test

def save_processed_data_files(X_train, X_test, y_train, y_test, output_dir):
    """
    Menyimpan data training dan testing yang sudah diproses ke file CSV terpisah.

    Args:
        X_train (pd.DataFrame): Fitur training.
        X_test (pd.DataFrame): Fitur testing.
        y_train (pd.Series): Target training.
        y_test (pd.Series): Target testing.
        output_dir (str): Direktori tempat file akan disimpan.
    """
    print("\n--- Memulai Penyimpanan Data yang Diproses ---")
    try:
        os.makedirs(output_dir, exist_ok=True) 

        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=True) 
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=True) 

        print(f"  Data yang diproses berhasil disimpan di: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan saat menyimpan data yang diproses: {e}")
    print("--- Penyimpanan Data Selesai ---")


# --- 3. Fungsi Utama untuk Orkesrasi Preprocessing ---

def run_telco_churn_preprocessing(raw_data_path, processed_output_dir):
    """
    Fungsi utama untuk mengorkestrasi seluruh alur preprocessing data Telco Churn.

    Args:
        raw_data_path (str): Path ke file CSV dataset mentah.
        processed_output_dir (str): Direktori untuk menyimpan data yang sudah diproses.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) jika preprocessing berhasil,
               atau (None, None, None, None) jika ada kesalahan.
    """
    print("\n##### Memulai Proses Preprocessing Data Telco Churn Otomatis #####")
    print(f"Lokasi direktori kerja saat skrip dijalankan: {os.getcwd()}")
    print(f"Path data mentah yang akan diakses: {os.path.abspath(raw_data_path)}")
    print(f"Direktori data terproses yang akan disimpan: {os.path.abspath(processed_output_dir)}")

    df = load_data(raw_data_path)
    if df is None:
        return None, None, None, None

    df = handle_total_charges(df)

    df = encode_binary_features(df, cols_to_encode=BINARY_COLS)

    df = encode_multi_class_features(df, cols_to_encode=MULTI_CLASS_COLS)

    df = handle_outliers_iqr_clip(df, columns_to_treat=NUMERIC_COLS_FOR_OUTLIERS)

    X_train, X_test, y_train, y_test = split_and_scale_data(
        df,
        target_column='Churn',
        cols_to_scale=NUMERIC_COLS_FOR_SCALING,
        id_column='customerID' 
    )
    if X_train is None:
        print("Proses split dan scaling gagal, menghentikan preprocessing.")
        return None, None, None, None

    save_processed_data_files(X_train, X_test, y_train, y_test, processed_output_dir)

    print("\n##### Proses Preprocessing Data Telco Churn Otomatis Selesai! #####")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train_final, X_test_final, y_train_final, y_test_final = run_telco_churn_preprocessing(
        raw_data_path=RAW_DATA_PATH,
        processed_output_dir=PROCESSED_DATA_DIR
    )

    if X_train_final is not None:
        print("\nVerifikasi hasil preprocessing:")
        print(f"  Shape X_train_final: {X_train_final.shape}")
        print(f"  Shape X_test_final: {X_test_final.shape}")
        print(f"  Shape y_train_final: {y_train_final.shape}")
        print(f"  Shape y_test_final: {y_test_final.shape}")

        # Verifikasi keberadaan file output
        if os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv')):
            print(f"  File X_train.csv berhasil dibuat di '{os.path.abspath(PROCESSED_DATA_DIR)}'.")
        else:
            print(f"  PERHATIAN: File X_train.csv TIDAK ditemukan setelah skrip dijalankan.")
    else:
        print("\nPreprocessing gagal atau tidak menghasilkan output yang valid.")

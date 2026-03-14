# ====================== mixed_data_preparation.py ======================
"""
Mixed data preprocessing script - modified version: keep only normalized _scaled columns
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# ==================== Configuration ====================
# Input: mixed data CSV file
MIXED_DATA_PATH = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\mixed_dataset_hourly_35040.csv"

# Output directory: create a separate preprocessing directory for mixed data
OUTPUT_DIR = r"D:\Oswaldo's surf project\70% Virtual 30% Real Database_Hourly\preprocessed_data_mixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Window parameters (consistent with original script)
WINDOW_SIZE = 24  # 24 hours = 1 day
STRIDE = 1        # sliding stride
PREDICTION_HORIZON = 0  # anomaly detection uses 0

# ==================== Feature column configuration ====================
# Original column names in the mixed data (non-normalized)
ORIGINAL_FEATURES = [
    'avg_PM2.5',
    'total_noise_duration',
    'noise_event_count',
    'avg_salience'
]

# Normalized column names (must have _scaled suffix)
SCALED_FEATURES = [
    'avg_PM2.5_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled'
]


# ==================== Main function ====================
def main():
    print("=" * 80)
    print("Mixed data preprocessing script (keep only _scaled columns)")
    print("=" * 80)

    # 1. Load mixed data
    print("\n1. Loading mixed data...")
    df = pd.read_csv(MIXED_DATA_PATH, parse_dates=['timestamp'])

    # Set timestamp as index
    df = df.set_index('timestamp')
    df = df.sort_index()

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    print("All columns:", df.columns.tolist())

    # Check existence of original features
    available_original_features = []
    missing_features = []

    for feat in ORIGINAL_FEATURES:
        if feat in df.columns:
            available_original_features.append(feat)
        else:
            missing_features.append(feat)
            print(f"⚠️ Warning: Feature {feat} not in data")

    if len(missing_features) > 0:
        print(f"\n🔍 Searching for possible alternative column names:")
        for missing_feat in missing_features:
            possible_matches = [col for col in df.columns if missing_feat in col]
            if possible_matches:
                print(f"  {missing_feat} → possible matches: {possible_matches}")
                # Use the first match
                if possible_matches[0] not in available_original_features:
                    available_original_features.append(possible_matches[0])
            else:
                print(f"  {missing_feat} → no matches found")

    print(f"\nAvailable original features: {available_original_features}")

    if len(available_original_features) < 2:
        print("❌ Error: Too few available features, please check column names")
        return

    # 2. Check missing values
    print("\n2. Checking missing values...")
    missing_counts = df[available_original_features].isna().sum()
    print(missing_counts)

    if missing_counts.sum() > 0:
        print(f"  Filling {missing_counts.sum()} missing values")
        df_filled = df[available_original_features].fillna(0)
    else:
        df_filled = df[available_original_features]

    # 3. ==================== Apply standardization and create _scaled columns ====================
    print("\n3. Standardizing mixed data (creating only _scaled columns)...")

    scalers = {}
    normalized_df = pd.DataFrame(index=df_filled.index)

    # For each original feature, apply standardization and create only the _scaled column
    for i, original_col in enumerate(available_original_features):
        scaler = StandardScaler()
        original_vals = df_filled[[original_col]].values

        # Apply standardization
        scaled_vals = scaler.fit_transform(original_vals)

        # Use predefined _scaled column name
        if i < len(SCALED_FEATURES):
            scaled_col = SCALED_FEATURES[i]
        else:
            scaled_col = f"{original_col}_scaled"

        # Add only the normalized column (do not keep the original column)
        normalized_df[scaled_col] = scaled_vals

        # Save the scaler
        scalers[original_col] = scaler

        print(f"  → Standardized {original_col} → {scaled_col}")
        print(f"     Original: mean={original_vals.mean():.4f}, std={original_vals.std():.4f}")
        print(f"     Standardized: mean={scaled_vals.mean():.6f}, std={scaled_vals.std():.6f}")

    # Obtain final feature column names (only _scaled columns)
    scaled_columns = normalized_df.columns.tolist()
    print(f"\nNormalized column names (only _scaled columns): {scaled_columns}")

    final_df = normalized_df[scaled_columns]  # ensure only _scaled columns are kept

    print("\nStatistics after standardization (only _scaled columns):")
    for col in scaled_columns:
        print(f"  {col}: mean={final_df[col].mean():.6f}, std={final_df[col].std():.6f}")

    # 4. Create sliding windows
    def create_sliding_windows(data, window_size, stride=1, horizon=0):
        """Create 3D sliding window array: (n_samples, window_size, n_features)"""
        X = []
        y = [] if horizon > 0 else None

        for i in range(0, len(data) - window_size - horizon + 1, stride):
            window = data[i: i + window_size]
            X.append(window)
            if horizon > 0:
                target = data[i + window_size: i + window_size + horizon]
                y.append(target)

        X = np.array(X)
        if y is not None:
            y = np.array(y)

        return X, y

    print(f"\n4. Creating sliding windows (window_size={WINDOW_SIZE}, stride={STRIDE})...")

    # Use normalized feature values (only columns with _scaled suffix)
    data_array = final_df[scaled_columns].values

    X_windows, y_windows = create_sliding_windows(
        data_array,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        horizon=PREDICTION_HORIZON
    )

    print(f"Window data shape: {X_windows.shape}")  # (n_windows, timesteps, n_features)

    if y_windows is not None:
        print(f"Target data shape: {y_windows.shape}")

    # 5. Save processed data
    print("\n5. Saving processed data...")

    # Save window data
    np.save(os.path.join(OUTPUT_DIR, "X_windows.npy"), X_windows)
    print(f"  → {OUTPUT_DIR}/X_windows.npy")

    if y_windows is not None:
        np.save(os.path.join(OUTPUT_DIR, "y_windows.npy"), y_windows)
        print(f"  → {OUTPUT_DIR}/y_windows.npy")

    # Save normalized hourly data (only _scaled columns)
    final_df.to_csv(os.path.join(OUTPUT_DIR, "normalized_hourly_data.csv"))
    print(f"  → {OUTPUT_DIR}/normalized_hourly_data.csv")
    print(f"  File contains columns: {final_df.columns.tolist()}")

    # Save scalers
    import joblib
    for col, scaler in scalers.items():
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"scaler_{col}.pkl"))
        print(f"  → {OUTPUT_DIR}/scaler_{col}.pkl")

    # Save all scalers in one file
    joblib.dump(scalers, os.path.join(OUTPUT_DIR, "all_scalers.pkl"))
    print(f"  → {OUTPUT_DIR}/all_scalers.pkl")

    # 6. Data statistics
    print("\n" + "=" * 80)
    print("Data preprocessing completed!")
    print("=" * 80)

    n_windows = X_windows.shape[0]
    n_hourly = final_df.shape[0]

    print(f"\n📊 Data statistics:")
    print(f"  Original hourly data: {n_hourly:,} hours")
    print(f"  Number of windows created: {n_windows:,} (24-hour windows)")
    print(f"  Window shape: {X_windows.shape}")
    print(f"  Number of features: {X_windows.shape[2]}")
    print(f"  Feature columns used (only _scaled columns): {scaled_columns}")

    # Window data statistics
    print(f"\n📊 Window data statistics:")
    print(f"  Min: {X_windows.min():.6f}")
    print(f"  Max: {X_windows.max():.6f}")
    print(f"  Mean: {X_windows.mean():.6f}")
    print(f"  Std: {X_windows.std():.6f}")

    # Create metadata file
    metadata = {
        "source_data": "mixed_dataset_hourly_35040.csv",
        "preprocessing_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hourly_samples": int(n_hourly),
        "window_samples": int(n_windows),
        "window_shape": list(X_windows.shape),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "original_features": available_original_features,
        "scaled_features": scaled_columns,
        "standardized": True,
        "scaler_info": {
            col: {
                "mean": float(scaler.mean_[0]),
                "scale": float(scaler.scale_[0]),
                "var": float(scaler.var_[0])
            } for col, scaler in scalers.items()
        },
        "data_statistics": {
            "min": float(X_windows.min()),
            "max": float(X_windows.max()),
            "mean": float(X_windows.mean()),
            "std": float(X_windows.std())
        },
        "output_files": [
                            "X_windows.npy",
                            "normalized_hourly_data.csv",
                            "all_scalers.pkl"
                        ] + [f"scaler_{col}.pkl" for col in scalers.keys()],
        "note": "normalized_hourly_data.csv contains only normalized _scaled columns, not original feature columns"
    }

    import json
    metadata_path = os.path.join(OUTPUT_DIR, "preprocessing_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Metadata file: {metadata_path}")

    # Show preview of CSV file content
    print("\n📋 Preview of normalized CSV file (first 5 rows, only _scaled columns):")
    print(final_df.head().to_string())

    print("\n📋 CSV file column names:")
    for i, col in enumerate(final_df.columns, 1):
        print(f"  {i}. {col}")

    print("\n✅ Preprocessing completed!")
    print(f"📍 Files saved in: {OUTPUT_DIR}")


# ==================== Run script ====================
if __name__ == "__main__":
    main()

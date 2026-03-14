import pandas as pd
import numpy as np
import os


def augment_to_100k_with_continuity_fixed(original_path, output_path, target_samples=100000):
    """
    Perform data augmentation within the normalized space (preserving z-score standardization characteristics)
    """
    print("🚀 Starting data augmentation (within z-score normalized space)...")

    # 1. Load original data
    df = pd.read_csv(original_path)
    print(f"📊 Original data shape: {df.shape}")
    print(f"📋 Column names: {df.columns.tolist()}")

    # 2. Check required columns
    scaled_features = ['avg_PM2.5_normalized_scaled', 'total_noise_duration_scaled',
                       'noise_event_count_scaled', 'avg_salience_scaled']

    for col in scaled_features:
        if col not in df.columns:
            raise ValueError(f"❌ Missing normalized column: {col}")

    if 'avg_PM2.5' not in df.columns:
        raise ValueError(f"❌ Missing original PM2.5 column: avg_PM2.5")

    # 3. Extract normalized features (z-score normalized, not min-max)
    X_scaled = df[scaled_features].values
    pm25_original = df['avg_PM2.5'].values

    print(f"📊 Normalized feature range verification (z-score normalization):")
    for i, col in enumerate(scaled_features):
        min_val = X_scaled[:, i].min()
        max_val = X_scaled[:, i].max()
        mean_val = X_scaled[:, i].mean()
        std_val = X_scaled[:, i].std()
        print(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")

    # 4. Generate target number of data points
    original_len = len(df)
    repeats_needed = int(np.ceil(target_samples / original_len)) + 2

    print(f"🔄 Original length: {original_len}, Target length: {target_samples}")
    print(f"🔄 Repeats needed: {repeats_needed}")

    # Store all augmented data
    all_augmented_scaled = []
    all_augmented_pm25 = []
    all_timestamps = []

    # Starting time
    current_time = pd.Timestamp('2013-03-01 00:00:00')

    # Different seasonal noise patterns
    season_patterns = [
        {'name': 'spring', 'base_noise': 0.1, 'trend_factor': 0.5},  # Adjusted noise level
        {'name': 'summer', 'base_noise': 0.15, 'trend_factor': 0.8},
        {'name': 'fall', 'base_noise': 0.08, 'trend_factor': 0.3},
        {'name': 'winter', 'base_noise': 0.2, 'trend_factor': 1.0}
    ]

    for i in range(repeats_needed):
        current_samples = len(np.concatenate(all_augmented_scaled)) if all_augmented_scaled else 0
        if current_samples >= target_samples:
            break

        # Get current season pattern
        season = season_patterns[i % len(season_patterns)]

        # Copy original normalized data
        chunk_scaled = X_scaled.copy()

        # Add noise within z-score normalized space (no clipping to 0-1 needed)
        noise = np.random.normal(0, season['base_noise'], chunk_scaled.shape)

        # Add slight time trend (simulating diurnal variation)
        time_trend = np.linspace(-0.1, 0.1, len(chunk_scaled)).reshape(-1, 1)
        time_trend = np.tile(time_trend, (1, chunk_scaled.shape[1]))

        noise += time_trend * season['trend_factor']

        # Apply noise (preserve z-score characteristics, no clipping)
        chunk_augmented = chunk_scaled + noise

        # For original PM2.5 values, adjust slightly based on augmented normalized values
        pm25_season_factor = 1.0 + 0.1 * np.sin(i * np.pi / 2)  # Seasonal factor
        pm25_chunk = pm25_original * pm25_season_factor

        # Add some randomness
        pm25_noise = np.random.normal(0, pm25_original.std() * 0.1, len(pm25_chunk))
        pm25_chunk = np.clip(pm25_chunk + pm25_noise, 0, None)  # PM2.5 cannot be negative

        # Generate timestamps (note: frequency is lowercase 'h', not uppercase 'H')
        chunk_timestamps = pd.date_range(
            start=current_time,
            periods=len(chunk_augmented),
            freq='h'  # Corrected: lowercase 'h'
        )
        current_time = chunk_timestamps[-1] + pd.Timedelta(hours=1)

        # Append to result lists
        all_augmented_scaled.append(chunk_augmented)
        all_augmented_pm25.append(pm25_chunk)
        all_timestamps.extend(chunk_timestamps)

        print(f"  🔄 Completed block {i + 1}, season: {season['name']}, current total samples: {current_samples + len(chunk_augmented)}")

    # 5. Merge all data
    print("🔗 Merging data blocks...")
    final_scaled = np.vstack(all_augmented_scaled)[:target_samples]
    final_pm25 = np.hstack(all_augmented_pm25)[:target_samples]
    final_timestamps = all_timestamps[:target_samples]

    # 6. Create final DataFrame
    augmented_df = pd.DataFrame(
        final_scaled,
        columns=scaled_features,
        index=final_timestamps
    )
    augmented_df['avg_PM2.5'] = final_pm25

    print(f"✅ Augmentation complete! Final shape: {augmented_df.shape}")
    print(f"⏰ Time range: {augmented_df.index[0]} to {augmented_df.index[-1]}")

    # 7. Validate augmented results
    print("\n📊 Augmented data validation:")
    print("Normalized feature statistics (z-score normalized):")
    for col in scaled_features:
        min_val = augmented_df[col].min()
        max_val = augmented_df[col].max()
        mean_val = augmented_df[col].mean()
        std_val = augmented_df[col].std()
        print(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")

    print(f"\nOriginal PM2.5 statistics:")
    print(f"  Min: {augmented_df['avg_PM2.5'].min():.2f}")
    print(f"  Max: {augmented_df['avg_PM2.5'].max():.2f}")
    print(f"  Mean: {augmented_df['avg_PM2.5'].mean():.2f}")
    print(f"  Std: {augmented_df['avg_PM2.5'].std():.2f}")
    print(f"  Negative value count: {(augmented_df['avg_PM2.5'] < 0).sum()}")

    # 8. Save data
    augmented_df.to_csv(output_path)
    print(f"💾 Data saved to: {output_path}")

    return augmented_df


def create_sliding_windows_simple(data, window_size=24, stride=1):
    """
    Create sliding windows (simplified version)
    """
    print(f"\n🎯 Creating sliding windows (window size={window_size}, stride={stride})...")

    # Use only normalized features
    scaled_features = ['avg_PM2.5_normalized_scaled', 'total_noise_duration_scaled',
                       'noise_event_count_scaled', 'avg_salience_scaled']

    X = data[scaled_features].values

    n_samples = len(X)
    n_features = X.shape[1]

    # Calculate number of windows
    n_windows = (n_samples - window_size) // stride + 1

    # Create windows
    windows = np.zeros((n_windows, window_size, n_features))

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = X[start:end]

    print(f"  Original samples: {n_samples}")
    print(f"  Number of features: {n_features}")
    print(f"  Number of windows created: {n_windows}")
    print(f"  Window shape: {windows.shape}")

    return windows


def create_and_save_training_data(augmented_df, output_dir):
    """
    Create and save training data
    """
    print("\n📁 Creating training dataset...")

    # 1. Create sliding windows
    X_windows_100k = create_sliding_windows_simple(augmented_df, window_size=24, stride=1)

    # 2. Split dataset (85% training, 15% validation)
    split_idx = int(len(X_windows_100k) * 0.85)
    X_train = X_windows_100k[:split_idx]
    X_val = X_windows_100k[split_idx:]

    print(f"\n📊 Dataset split:")
    print(f"  Training set: {X_train.shape[0]} windows ({split_idx / len(X_windows_100k) * 100:.1f}%)")
    print(f"  Validation set: {X_val.shape[0]} windows ({len(X_val) / len(X_windows_100k) * 100:.1f}%)")

    # 3. Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 4. Save data
    np.save(os.path.join(output_dir, "X_windows_100k.npy"), X_windows_100k)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)

    print(f"\n💾 Files saved to: {output_dir}")
    print(f"  - X_windows_100k.npy: shape {X_windows_100k.shape}")
    print(f"  - X_train.npy: shape {X_train.shape}")
    print(f"  - X_val.npy: shape {X_val.shape}")

    return X_windows_100k, X_train, X_val


# ====================== Main Program ======================
if __name__ == "__main__":
    # Configure file paths
    ORIGINAL_DATA_PATH = r"D:\Oswaldo's surf project\DR O's database\preprocessed_data\normalized_hourly_data.csv"
    OUTPUT_DIR = r"D:\Oswaldo's surf project\DR O's database\preprocessed_data"
    AUGMENTED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "augmented_100k_fixed.csv")

    print("=" * 60)
    print("🚀 100k Data Augmentation System (within z-score normalized space)")
    print("=" * 60)

    try:
        # Step 1: Generate 100k augmented data points
        print("\n📦 Step 1: Generating augmented data")
        augmented_data = augment_to_100k_with_continuity_fixed(
            original_path=ORIGINAL_DATA_PATH,  # Explicitly pass parameters
            output_path=AUGMENTED_OUTPUT_PATH,
            target_samples=100000
        )

        # Step 2: Create training windows
        print("\n🎯 Step 2: Creating training windows")
        X_windows_100k, X_train, X_val = create_and_save_training_data(
            augmented_df=augmented_data,  # Explicitly pass parameters
            output_dir=OUTPUT_DIR
        )

        # Step 3: Data validation
        print("\n🔍 Step 3: Data validation")
        print("Window data statistics:")
        print(f"  X_windows_100k shape: {X_windows_100k.shape}")
        print(f"  X_windows_100k mean: {X_windows_100k.mean():.4f}")
        print(f"  X_windows_100k std: {X_windows_100k.std():.4f}")
        print(f"  X_windows_100k range: [{X_windows_100k.min():.4f}, {X_windows_100k.max():.4f}]")

        print(f"\nData integrity check:")
        print(f"  NaN count: {np.isnan(X_windows_100k).sum()}")
        print(f"  Infinite value count: {np.isinf(X_windows_100k).sum()}")

        # Step 4: Usage guide
        print("\n" + "=" * 60)
        print("📋 Usage Guide")
        print("=" * 60)
        print("✅ Data augmentation complete!")
        print("\n📁 Generated files:")
        print(f"  1. {AUGMENTED_OUTPUT_PATH} - Augmented full dataset")
        print(f"  2. {OUTPUT_DIR}/X_windows_100k.npy - Sliding window data")
        print(f"  3. {OUTPUT_DIR}/X_train.npy - Training set")
        print(f"  4. {OUTPUT_DIR}/X_val.npy - Validation set")

        print("\n🔧 How to use:")
        print("1. Load data:")
        print("   X_windows_100k = np.load('X_windows_100k.npy')")
        print("2. Train AutoEncoder:")
        print("   model.fit(X_train, X_train, validation_data=(X_val, X_val))")
        print("3. Features are z-score normalized (mean ~0, std ~1)")
        print("4. Time series continuity preserved")

        print("\n📊 Data statistics:")
        print(f"  Total samples: {len(augmented_data)}")
        print(f"  Total windows: {len(X_windows_100k)}")
        print(f"  Training windows: {len(X_train)} ({len(X_train) / len(X_windows_100k) * 100:.1f}%)")
        print(f"  Validation windows: {len(X_val)} ({len(X_val) / len(X_windows_100k) * 100:.1f}%)")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()

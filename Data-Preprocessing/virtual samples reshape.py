import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwise('ignore')


# ====================== Core Function - Hierarchical Pattern Extraction (modified to 4 features) ======================
def extract_hierarchical_patterns(df_aircraft, df_pm25):
    """
    Extract hierarchical pattern features (global + 24-hour level + time period level) - only extract 4 core features
    :param df_aircraft: Preprocessed aircraft noise data
    :param df_pm25: Preprocessed PM2.5 data
    :return: Hierarchical pattern dictionary
    """
    patterns = {}
    pm25_city_cols = [col for col in df_pm25.columns if col != 'timestamp']  # All city PM2.5 columns

    # First layer: global statistical features (reduced to 4 core)
    patterns['global'] = {
        'total_aircraft_records': len(df_aircraft),
        'overall_noise_mean': df_aircraft['max_slow'].mean() if len(df_aircraft) > 0 else 0,
        'overall_pm25_mean': df_pm25[pm25_city_cols].values.flatten().mean() if len(df_pm25) > 0 else 0,
        'aircraft_type_diversity': df_aircraft['type'].nunique() / len(df_aircraft) if len(df_aircraft) > 0 else 0
    }

    # Second layer: 24-hour pattern features (only extract 4 core features)
    for hour in range(24):
        aircraft_hour = df_aircraft[df_aircraft['timestamp'].dt.hour == hour]
        pm25_hour = df_pm25[df_pm25['timestamp'].dt.hour == hour]
        hour_features = {}

        # Only extract 4 core features (aligned with real data)
        # 1. PM2.5 feature (corresponds to avg_PM2.5 in real data)
        if len(pm25_hour) > 0:
            pm25_hour_values = pm25_hour[pm25_city_cols].values.flatten()
            hour_features['pm25_mean'] = pm25_hour_values.mean()
        else:
            hour_features['pm25_mean'] = 0

        # 2. Aircraft noise feature (corresponds to total_noise_duration in real data)
        hour_features['noise_mean'] = aircraft_hour['max_slow'].mean() if len(aircraft_hour) > 0 else 0

        # 3. Aircraft count feature (corresponds to noise_event_count in real data)
        hour_features['aircraft_count'] = len(aircraft_hour)

        # 4. Aircraft type diversity feature (corresponds to avg_salience in real data)
        hour_features['aircraft_type_diversity'] = aircraft_hour['type'].nunique() / len(aircraft_hour) if len(
            aircraft_hour) > 0 else 0

        patterns[f'hour_{hour}'] = hour_features

    # Third layer: time period pattern features (only extract 4 core features)
    time_windows = {'morning': (6, 10), 'daytime': (10, 17), 'evening': (17, 21), 'night': (21, 6)}
    for period, (start, end) in time_windows.items():
        period_features = {}

        # Aircraft period filtering
        if start < end:
            aircraft_mask = (df_aircraft['timestamp'].dt.hour >= start) & (df_aircraft['timestamp'].dt.hour < end)
        else:
            aircraft_mask = (df_aircraft['timestamp'].dt.hour >= start) | (df_aircraft['timestamp'].dt.hour < end)
        aircraft_period = df_aircraft[aircraft_mask]

        # PM2.5 period filtering
        if start < end:
            pm25_mask = (df_pm25['timestamp'].dt.hour >= start) & (df_pm25['timestamp'].dt.hour < end)
        else:
            pm25_mask = (df_pm25['timestamp'].dt.hour >= start) | (df_pm25['timestamp'].dt.hour < end)
        pm25_period = df_pm25[pm25_mask]

        # Only extract 4 core features
        period_features['aircraft_count'] = len(aircraft_period)
        period_features['noise_mean'] = aircraft_period['max_slow'].mean() if len(aircraft_period) > 0 else 0
        period_features['pm25_mean'] = pm25_period[pm25_city_cols].values.flatten().mean() if len(
            pm25_period) > 0 else 0
        period_features['aircraft_type_diversity'] = aircraft_period['type'].nunique() / len(aircraft_period) if len(
            aircraft_period) > 0 else 0

        patterns[f'period_{period}'] = period_features

    print("✅ Hierarchical pattern feature extraction completed (4 core features: PM2.5 mean, noise mean, aircraft count, aircraft type diversity)")
    return patterns


# ====================== Pattern Features to DAE Input (modified to 4 features) ======================
def patterns_to_dae_features(patterns_dict):
    """Convert to standardized features usable by DeepAutoEncoder (4-feature version)"""
    # 4 core features
    hour_feature_names = ['pm25_mean', 'noise_mean', 'aircraft_count', 'aircraft_type_diversity']

    # 24-hour feature matrix (core) - 4 features per hour
    hour_matrix = np.array([[patterns_dict[f'hour_{h}'][feat] for feat in hour_feature_names] for h in range(24)])

    # Global + time period feature vector (simplified)
    global_feats = [
        patterns_dict['global']['total_aircraft_records'],
        patterns_dict['global']['overall_noise_mean'],
        patterns_dict['global']['overall_pm25_mean'],
        patterns_dict['global']['aircraft_type_diversity']
    ]

    period_feats = []
    for p in ['morning', 'daytime', 'evening', 'night']:
        period_feats.extend([
            patterns_dict[f'period_{p}']['aircraft_count'],
            patterns_dict[f'period_{p}']['noise_mean'],
            patterns_dict[f'period_{p}']['pm25_mean'],
            patterns_dict[f'period_{p}']['aircraft_type_diversity']
        ])

    global_period_vector = np.array(global_feats + period_feats).reshape(1, -1)

    # Standardization - standardize hour features and global features separately
    hour_scaler = MinMaxScaler(feature_range=(0, 1))
    global_scaler = MinMaxScaler(feature_range=(0, 1))

    hour_matrix_scaled = hour_scaler.fit_transform(hour_matrix)
    global_period_scaled = global_scaler.fit_transform(global_period_vector)

    # Feature name mapping
    feature_mapping = {
        'hour_feature_names': hour_feature_names,
        'hour_feature_descriptions': {
            'pm25_mean': 'PM2.5 mean (corresponds to avg_PM2.5 in real data)',
            'noise_mean': 'Noise mean (corresponds to total_noise_duration in real data)',
            'aircraft_count': 'Aircraft count (corresponds to noise_event_count in real data)',
            'aircraft_type_diversity': 'Aircraft type diversity (corresponds to avg_salience in real data)'
        },
        'global_feature_names': ['total_aircraft_records', 'overall_noise_mean', 'overall_pm25_mean',
                                 'aircraft_type_diversity'],
        'period_feature_names': [f'period_{p}_{f}' for p in ['morning', 'daytime', 'evening', 'night']
                                 for f in ['aircraft_count', 'noise_mean', 'pm25_mean', 'aircraft_type_diversity']],
        'real_data_features': ['avg_PM2.5', 'total_noise_duration', 'noise_event_count', 'avg_salience']
    }

    print("✅ 4-feature conversion completed:")
    print(f"  - 24-hour feature matrix shape: {hour_matrix_scaled.shape} (24 hours × 4 features)")
    print(f"  - Global + time period feature vector shape: {global_period_scaled.shape}")
    print(f"  - Feature correspondence:")
    print(f"     Virtual data: {hour_feature_names}")
    print(f"     Real data: {feature_mapping['real_data_features']}")

    return hour_matrix_scaled, global_period_scaled, feature_mapping


# ====================== Generate 4-feature Virtual Data Samples (new function) ======================
def generate_4feature_virtual_samples(hour_matrix_scaled, n_samples=35040, variation_factor=0.1):
    """
    Generate 4-feature virtual data samples based on extracted patterns

    Parameters:
        hour_matrix_scaled: Standardized hourly feature matrix (24, 4)
        n_samples: Number of samples to generate
        variation_factor: Variation factor controlling sample diversity
    """
    print(f"\n🔄 Generating 4-feature virtual data samples (n={n_samples})...")

    # Base pattern
    base_pattern = hour_matrix_scaled  # (24, 4)

    # Generate varied samples
    virtual_samples = []
    for i in range(n_samples):
        # Add random variation to each sample
        variation = np.random.normal(0, variation_factor, base_pattern.shape)
        sample = base_pattern + variation

        # Ensure values are within reasonable range
        sample = np.clip(sample, 0, 1)
        virtual_samples.append(sample)

    virtual_samples = np.array(virtual_samples)  # (n_samples, 24, 4)
    print(f"  Generated virtual data shape: {virtual_samples.shape}")

    return virtual_samples


# ====================== Main Process (directly read preprocessed data) ======================
if __name__ == "__main__":
    # ---------------------- Key: read your preprocessed files ----------------------
    # Please confirm the following paths are the actual paths to your preprocessed files, modify and run!
    processed_pm25_path = r"D:\Oswaldo's surf project\My Database\PM2.5\random_city3PM25_data.csv"
    processed_noise_path = r"D:\Oswaldo's surf project\My Database\aircraft noise_data_test4.csv"

    # Read data (already preprocessed, load directly)
    pm25_df = pd.read_csv(processed_pm25_path)
    noise_df = pd.read_csv(processed_noise_path)

    # Convert timestamp column format (ensure dt.hour is available, must keep)
    pm25_df['timestamp'] = pd.to_datetime(pm25_df['timestamp'])
    noise_df['timestamp'] = pd.to_datetime(noise_df['timestamp'])

    print(f"✅ Preprocessed data loaded:")
    print(f"  - PM2.5 data: {pm25_df.shape} records, missing values: {pm25_df.isnull().sum().sum()}")
    print(f"  - Aircraft noise data: {noise_df.shape} records, missing values: {noise_df.isnull().sum().sum()}")

    # Extract pattern features (4-feature version)
    hierarchical_patterns = extract_hierarchical_patterns(noise_df, pm25_df)

    # Convert to DAE features and save (4-feature version)
    hour_matrix_scaled, global_period_scaled, feature_mapping = patterns_to_dae_features(hierarchical_patterns)

    # Save results (path can be modified)
    save_path = r"D:\Oswaldo's surf project\My Database\Merge_and_DAE_4_extracted_features"
    import os

    os.makedirs(save_path, exist_ok=True)

    # Save 4-feature version files
    np.save(os.path.join(save_path, "24hour_pattern_matrix_4features_scaled.npy"), hour_matrix_scaled)
    np.save(os.path.join(save_path, "global_period_pattern_vector_4features_scaled.npy"), global_period_scaled)

    # Generate and save 4-feature virtual data samples
    virtual_samples = generate_4feature_virtual_samples(hour_matrix_scaled, n_samples=35040)
    np.save(os.path.join(save_path, "virtual_samples_4features_35040.npy"), virtual_samples)

    # Save feature mapping
    import json

    with open(os.path.join(save_path, "feature_mapping_4features.json"), 'w', encoding='utf-8') as f:
        json.dump(feature_mapping, f, ensure_ascii=False, indent=2)

    # Print verification information
    print("\n📊 Example: 0:00 pattern features (4 features, standardized):")
    print(f"  - PM2.5 mean: {hour_matrix_scaled[0][0]:.3f}")
    print(f"  - Noise mean: {hour_matrix_scaled[0][1]:.3f}")
    print(f"  - Aircraft count: {hour_matrix_scaled[0][2]:.3f}")
    print(f"  - Aircraft type diversity: {hour_matrix_scaled[0][3]:.3f}")

    # Feature statistics
    print("\n📈 4-feature statistics:")
    for i, feat_name in enumerate(feature_mapping['hour_feature_names']):
        feat_data = hour_matrix_scaled[:, i]
        print(
            f"  {feat_name}: mean={feat_data.mean():.3f}, std={feat_data.std():.3f}, range=[{feat_data.min():.3f}, {feat_data.max():.3f}]")

    # Virtual data quality check
    print("\n🔍 Virtual data quality check:")
    print(f"  Virtual data shape: {virtual_samples.shape}")
    print(f"  Virtual data range: [{virtual_samples.min():.3f}, {virtual_samples.max():.3f}]")
    print(f"  Virtual data mean: {virtual_samples.mean():.3f} ± {virtual_samples.std():.3f}")
    print(f"  Missing values: {np.isnan(virtual_samples).sum()}")

    print(f"\n🎉 4-feature version run completed! Files saved to: {save_path}")
    print("📁 Generated files:")
    print(f"  1. 24hour_pattern_matrix_4features_scaled.npy - 24-hour 4-feature pattern matrix")
    print(f"  2. global_period_pattern_vector_4features_scaled.npy - Global + time period feature vector")
    print(f"  3. virtual_samples_4features_35040.npy - 4-feature virtual data samples (35040 samples)")
    print(f"  4. feature_mapping_4features.json - Feature mapping file")
    print("\n💡 Now you can directly use these 4-feature files for subsequent mixed dataset creation!")

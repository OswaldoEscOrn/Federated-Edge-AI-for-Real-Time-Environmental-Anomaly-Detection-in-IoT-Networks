
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ====================== Core Function - Hierarchical Pattern Extraction (Scheme 3 core, no modifications) ======================
def extract_hierarchical_patterns(df_aircraft, df_pm25):
    """
    Extract hierarchical pattern features (global + 24-hour level + time period level)
    :param df_aircraft: Preprocessed aircraft noise data
    :param df_pm25: Preprocessed PM2.5 data
    :return: Hierarchical pattern dictionary
    """
    patterns = {}
    pm25_city_cols = [col for col in df_pm25.columns if col != 'timestamp']  # All city PM2.5 columns

    # First layer: global statistical features
    patterns['global'] = {
        'total_aircraft_records': len(df_aircraft),
        'overall_noise_mean': df_aircraft['max_slow'].mean() if len(df_aircraft) > 0 else 0,
        'overall_noise_std': df_aircraft['max_slow'].std() if len(df_aircraft) > 1 else 0,
        'B738_global_ratio': (df_aircraft['type'] == 'B738').mean() if len(df_aircraft) > 0 else 0,
        'total_pm25_records': len(df_pm25),
        'overall_pm25_mean': df_pm25[pm25_city_cols].values.flatten().mean() if len(df_pm25) > 0 else 0,
        'overall_pm25_std': df_pm25[pm25_city_cols].values.flatten().std() if len(df_pm25) > 1 else 0,
        'city_pm25_variance': df_pm25[pm25_city_cols].var(axis=1).mean() if len(df_pm25) > 0 else 0
    }

    # Second layer: 24-hour pattern features (core)
    for hour in range(24):
        aircraft_hour = df_aircraft[df_aircraft['timestamp'].dt.hour == hour]
        pm25_hour = df_pm25[df_pm25['timestamp'].dt.hour == hour]
        hour_features = {}

        # Aircraft features for the hour
        hour_features['aircraft_count'] = len(aircraft_hour)
        hour_features['noise_mean'] = aircraft_hour['max_slow'].mean() if len(aircraft_hour) > 0 else 0
        hour_features['noise_std'] = aircraft_hour['max_slow'].std() if len(aircraft_hour) > 1 else 0
        hour_features['B738_ratio'] = (aircraft_hour['type'] == 'B738').mean() if len(aircraft_hour) > 0 else 0
        hour_features['A319_ratio'] = (aircraft_hour['type'] == 'A319').mean() if len(aircraft_hour) > 0 else 0
        hour_features['aircraft_type_diversity'] = aircraft_hour['type'].nunique() / len(aircraft_hour) if len(aircraft_hour) > 0 else 0

        # PM2.5 features for the hour
        if len(pm25_hour) > 0:
            pm25_hour_values = pm25_hour[pm25_city_cols].values.flatten()
            hour_features['pm25_mean'] = pm25_hour_values.mean()
            hour_features['pm25_std'] = pm25_hour_values.std() if len(pm25_hour_values) > 1 else 0
            hour_features['pm25_max'] = pm25_hour_values.max()
            hour_features['pm25_min'] = pm25_hour_values.min()
            hour_features['city_pm25_variance'] = pm25_hour[pm25_city_cols].var(axis=1).mean()
        else:
            hour_features['pm25_mean'] = hour_features['pm25_std'] = hour_features['pm25_max'] = hour_features['pm25_min'] = hour_features['city_pm25_variance'] = 0

        patterns[f'hour_{hour}'] = hour_features

    # Third layer: time period pattern features
    time_windows = {'morning': (6,10), 'daytime': (10,17), 'evening': (17,21), 'night': (21,6)}
    for period, (start, end) in time_windows.items():
        period_features = {}

        # Aircraft period filtering
        if start < end:
            aircraft_mask = (df_aircraft['timestamp'].dt.hour >= start) & (df_aircraft['timestamp'].dt.hour < end)
        else:
            aircraft_mask = (df_aircraft['timestamp'].dt.hour >= start) | (df_aircraft['timestamp'].dt.hour < end)
        aircraft_period = df_aircraft[aircraft_mask]
        period_features['aircraft_count'] = len(aircraft_period)
        period_features['noise_mean'] = aircraft_period['max_slow'].mean() if len(aircraft_period) > 0 else 0
        period_features['B738_ratio'] = (aircraft_period['type'] == 'B738').mean() if len(aircraft_period) > 0 else 0

        # PM2.5 period filtering
        if start < end:
            pm25_mask = (df_pm25['timestamp'].dt.hour >= start) & (df_pm25['timestamp'].dt.hour < end)
        else:
            pm25_mask = (df_pm25['timestamp'].dt.hour >= start) | (df_pm25['timestamp'].dt.hour < end)
        pm25_period = df_pm25[pm25_mask]
        period_features['pm25_mean'] = pm25_period[pm25_city_cols].values.flatten().mean() if len(pm25_period) > 0 else 0

        patterns[f'period_{period}'] = period_features

    print("✅ Hierarchical pattern feature extraction completed (global + 24 hours + 4 time periods)")
    return patterns

# ====================== Convert Pattern Features to DAE Input (no modifications) ======================
def patterns_to_dae_features(patterns_dict):
    """Convert to standardized features usable by DeepAutoEncoder"""
    # 24-hour feature matrix (core)
    hour_feature_names = ['aircraft_count', 'noise_mean', 'noise_std', 'B738_ratio', 'A319_ratio',
                         'aircraft_type_diversity', 'pm25_mean', 'pm25_std', 'pm25_max', 'pm25_min', 'city_pm25_variance']
    hour_matrix = np.array([[patterns_dict[f'hour_{h}'][feat] for feat in hour_feature_names] for h in range(24)])

    # Global + time period feature vector
    global_feats = [patterns_dict['global'][f] for f in ['total_aircraft_records', 'overall_noise_mean', 'overall_noise_std', 'B738_global_ratio',
                                                     'total_pm25_records', 'overall_pm25_mean', 'overall_pm25_std', 'city_pm25_variance']]
    period_feats = []
    for p in ['morning', 'daytime', 'evening', 'night']:
        period_feats.extend([patterns_dict[f'period_{p}']['aircraft_count'], patterns_dict[f'period_{p}']['noise_mean'],
                           patterns_dict[f'period_{p}']['B738_ratio'], patterns_dict[f'period_{p}']['pm25_mean']])
    global_period_vector = np.array(global_feats + period_feats).reshape(1, -1)

    # Standardization
    scaler = MinMaxScaler(feature_range=(0,1))
    hour_matrix_scaled = scaler.fit_transform(hour_matrix)
    global_period_scaled = scaler.fit_transform(global_period_vector)

    # Feature name mapping
    feature_mapping = {
        'hour_feature_names': hour_feature_names,
        'global_feature_names': ['total_aircraft_records', 'overall_noise_mean', 'overall_noise_std', 'B738_global_ratio',
                               'total_pm25_records', 'overall_pm25_mean', 'overall_pm25_std', 'city_pm25_variance'],
        'period_feature_names': [f'period_{p}_{f}' for p in ['morning', 'daytime', 'evening', 'night'] for f in ['aircraft_count', 'noise_mean', 'B738_ratio', 'pm25_mean']]
    }

    print("✅ Feature conversion completed:")
    print(f"  - 24-hour feature matrix shape: {hour_matrix_scaled.shape}")
    print(f"  - Global + time period feature vector shape: {global_period_scaled.shape}")
    return hour_matrix_scaled, global_period_scaled, feature_mapping

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

    # Extract pattern features
    hierarchical_patterns = extract_hierarchical_patterns(noise_df, pm25_df)

    # Convert to DAE features and save
    hour_matrix_scaled, global_period_scaled, feature_mapping = patterns_to_dae_features(hierarchical_patterns)

    # Save results (path can be modified)
    save_path = r"D:\Oswaldo's surf project\My Database\Merge_and_DAE"
    np.save(save_path + "24hour_pattern_matrix_scaled.npy", hour_matrix_scaled)
    np.save(save_path + "global_period_pattern_vector_scaled.npy", global_period_scaled)
    import json
    with open(save_path + "feature_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(feature_mapping, f, ensure_ascii=False, indent=2)

    # Print verification information
    print("\n📊 Example: 0:00 pattern features (standardized):")
    print(f"  - 0:00 average noise: {hour_matrix_scaled[0][1]:.3f}")
    print(f"  - 0:00 PM2.5 mean: {hour_matrix_scaled[0][6]:.3f}")
    print(f"  - 0:00 B738 ratio: {hour_matrix_scaled[0][3]:.3f}")

    print("\n🎉 Run completed! 3 files saved to the specified path, ready for DAE training~")

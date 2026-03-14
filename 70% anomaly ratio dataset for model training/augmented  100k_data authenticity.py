import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def environmental_augmentation_validation(original_df, augmented_df):
    """
    Augmentation quality validation for environmental monitoring data
    Features: avg_PM2.5_normalized, total_noise_duration_scaled,
              noise_event_count_scaled, avg_salience_scaled, avg_PM2.5
    """

    print("=" * 60)
    print("Environmental Monitoring Data Augmentation Quality Validation Report")
    print("=" * 60)

    results = {}

    # 1. Basic feature list
    features = ['avg_PM2.5_normalized_scaled', 'total_noise_duration_scaled',
                'noise_event_count_scaled', 'avg_salience_scaled', 'avg_PM2.5']

    scaled_features = ['avg_PM2.5_normalized_scaled', 'total_noise_duration_scaled',
                       'noise_event_count_scaled', 'avg_salience_scaled']

    # 2. Physical constraint check (most important!)
    print("\n1. Physical constraint check:")

    # PM2.5 cannot be negative
    pm25_negative = augmented_df['avg_PM2.5'] < 0
    if pm25_negative.any():
        print(f"  ⚠ Warning: {pm25_negative.sum()} records with negative PM2.5 values")
    else:
        print(f"  ✓ All PM2.5 values are non-negative")

    # Standardized features should be within reasonable range
    for feat in scaled_features:
        outliers = augmented_df[feat].abs() > 5  # Standardized usually should be in [-3,3]
        if outliers.any():
            print(f"  ⚠ {feat}: {outliers.sum()} values exceed ±5 range")

    # 3. Distribution characteristics validation (for different feature types)
    print("\n2. Distribution characteristics validation:")

    # PM2.5 usually follows log-normal distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        # Original data distribution
        axes[i].hist(original_df[feat].dropna(), bins=50, alpha=0.5,
                     label='original', density=True, color='blue')
        # Augmented data distribution
        axes[i].hist(augmented_df[feat].dropna(), bins=50, alpha=0.5,
                     label='augmented', density=True, color='red')

        # Statistical test
        ks_stat, ks_p = stats.ks_2samp(
            original_df[feat].dropna(),
            augmented_df[feat].dropna()
        )

        # Check skewness and kurtosis
        orig_skew = stats.skew(original_df[feat].dropna())
        aug_skew = stats.skew(augmented_df[feat].dropna())
        skew_diff = abs(orig_skew - aug_skew)

        axes[i].set_title(f'{feat}\nKS-p={ks_p:.3f}, skew={skew_diff:.2f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete\100k_environmental_dist_comparison.png", dpi=150)
    plt.close()

    # 4. Correlation preservation validation (environmental indicators usually have correlations)
    print("\n3. Inter-feature correlation validation:")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Original data correlation
    orig_corr = original_df[features].corr()
    sns.heatmap(orig_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax1)
    ax1.set_title('Original Data Correlation')

    # Augmented data correlation
    aug_corr = augmented_df[features].corr()
    sns.heatmap(aug_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax2)
    ax2.set_title('Augmented Data Correlation')

    plt.tight_layout()
    plt.savefig(r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete\100k_augmented_data_correlation.png", dpi=150)
    plt.close()

    # Compute correlation difference
    corr_diff = (orig_corr - aug_corr).abs()
    max_diff = corr_diff.max().max()
    mean_diff = corr_diff.mean().mean()

    print(f"  Maximum correlation difference: {max_diff:.3f}")
    print(f"  Mean correlation difference: {mean_diff:.3f}")

    if mean_diff < 0.1:
        print(f"  ✓ Correlation structure well preserved")
    else:
        print(f"  ⚠ Warning: Significant change in inter-variable relationships")

    # 5. Time series pattern check (if temporal information available)
    print("\n4. Time series pattern check (if applicable):")

    # (Note: Implementation depends on data, placeholder here)
    print("  (Not implemented, relies on actual temporal index)")

    # 6. Multivariate anomaly detection (detect anomalous patterns in augmented data)
    print("\n5. Multivariate pattern consistency check:")

    # Train isolation forest on original data
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(original_df[features])

    # Predict anomalies in augmented data
    aug_predictions = iso_forest.predict(augmented_df[features])
    anomaly_rate = (aug_predictions == -1).mean()

    print(f"  Proportion of augmented data flagged as anomalies by original data model: {anomaly_rate:.2%}")

    if anomaly_rate < 0.1:
        print(f"  ✓ Augmented data patterns consistent with original")
    elif anomaly_rate < 0.2:
        print(f"  ⚠ Note: {anomaly_rate:.1%} of augmented data differ from original pattern")
    else:
        print(f"  ⚠ Warning: Large proportion ({anomaly_rate:.1%}) of augmented data inconsistent with original pattern")

    # 7. Cluster structure preservation
    print("\n6. Cluster structure check:")

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    # Cluster original data
    kmeans_orig = KMeans(n_clusters=3, random_state=42)
    orig_labels = kmeans_orig.fit_predict(original_df[features].fillna(0))

    # Use same cluster centers for augmented data
    aug_labels = kmeans_orig.predict(augmented_df[features].fillna(0))

    # For comparison, also cluster augmented data separately
    kmeans_aug = KMeans(n_clusters=3, random_state=42)
    aug_labels_separate = kmeans_aug.fit_predict(augmented_df[features].fillna(0))

    # Compare cluster centers
    center_diff = np.abs(kmeans_orig.cluster_centers_ - kmeans_aug.cluster_centers_).mean()
    print(f"  Average difference in cluster centers: {center_diff:.3f}")

    # 8. Generate comprehensive quality report
    print("\n" + "=" * 60)
    print("Comprehensive Quality Score")
    print("=" * 60)

    # Calculate individual scores
    scores = {
        'Physical constraints': 1.0 if not pm25_negative.any() else 0.5,
        'Distribution consistency': 1.0 if mean_diff < 0.1 else 0.7 if mean_diff < 0.2 else 0.4,
        'Correlation preservation': 1.0 if mean_diff < 0.1 else 0.6,
        'Anomaly rate': 1.0 if anomaly_rate < 0.1 else 0.8 if anomaly_rate < 0.2 else 0.5,
        'Cluster structure': 1.0 if center_diff < 0.2 else 0.7
    }

    overall_score = np.mean(list(scores.values()))

    print(f"\nIndividual scores:")
    for key, value in scores.items():
        print(f"  {key}: {value:.2f}")

    print(f"\nOverall score: {overall_score:.2f}/1.0")

    if overall_score >= 0.9:
        rating = "Excellent - Ready for use"
    elif overall_score >= 0.7:
        rating = "Good - Recommend light inspection"
    elif overall_score >= 0.5:
        rating = "Moderate - Further validation needed"
    else:
        rating = "Needs improvement - Consider regenerating"

    print(f"Assessment: {rating}")

    # Save detailed results
    results = {
        'features': features,
        'physical_constraints_violated': pm25_negative.sum(),
        'correlation_mean_diff': mean_diff,
        'anomaly_rate_in_augmented': anomaly_rate,
        'cluster_center_diff': center_diff,
        'scores': scores,
        'overall_score': overall_score,
        'rating': rating
    }

    return results


# Usage example
# ====================== Main Program ======================
def main():
    # Load data
    original_df = pd.read_csv(r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete\preprocessed_time_series.csv")
    augmented_df = pd.read_csv(r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete\preprocessed_time_series_augmented_100k.csv")

    print("📊 Original data columns:", original_df.columns.tolist())
    print("📊 Augmented data columns:", augmented_df.columns.tolist())

    # Handle index columns
    if 'Unnamed: 0' in original_df.columns:
        print("🔄 Removing index column from original data...")
        original_df = original_df.drop(columns=['Unnamed: 0'])

    if 'Unnamed: 0' in augmented_df.columns:
        print("🔄 Removing index column from augmented data...")
        augmented_df = augmented_df.drop(columns=['Unnamed: 0'])

    # Verify required features exist
    required_features = ['avg_PM2.5_normalized_scaled', 'total_noise_duration_scaled',
                         'noise_event_count_scaled', 'avg_salience_scaled', 'avg_PM2.5']

    missing_original = [col for col in required_features if col not in original_df.columns]
    missing_augmented = [col for col in required_features if col not in augmented_df.columns]

    if missing_original:
        print(f"❌ Original data missing columns: {missing_original}")
        return

    if missing_augmented:
        print(f"❌ Augmented data missing columns: {missing_augmented}")
        return

    print("✅ All required feature columns present")
    print(f"📈 Original data shape: {original_df.shape}")
    print(f"📈 Augmented data shape: {augmented_df.shape}")

    # Run validation
    results = environmental_augmentation_validation(original_df, augmented_df)

    # Save results to file
    report_path = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_70%_anomaly_complete\100k_environmental_augmentation_validation_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Environmental Monitoring Data Augmentation Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Validation time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original data samples: {len(original_df)}\n")
        f.write(f"Augmented data samples: {len(augmented_df)}\n\n")

        for key, value in results.items():
            if key == 'features':
                f.write("Validated features:\n")
                for feat in value:
                    f.write(f"  - {feat}\n")
            elif key == 'scores':
                f.write("\nIndividual scores:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"💾 Detailed report saved to: {report_path}")

    # Provide recommendation based on score
    overall_score = results.get('overall_score', 0)
    print("\n" + "=" * 60)
    print("🎯 Validation Summary")
    print("=" * 60)

    if overall_score >= 0.9:
        print("✅ Excellent: Data augmentation quality is very high, ready for model training")
    elif overall_score >= 0.7:
        print("✅ Good: Data augmentation quality is good, recommend checking distribution plots before use")
    elif overall_score >= 0.5:
        print("⚠️ Moderate: Data augmentation quality is average, recommend regenerating or manual adjustment")
    else:
        print("❌ Needs improvement: Data augmentation quality is poor, recommend redesigning augmentation strategy")


if __name__ == "__main__":
    main()

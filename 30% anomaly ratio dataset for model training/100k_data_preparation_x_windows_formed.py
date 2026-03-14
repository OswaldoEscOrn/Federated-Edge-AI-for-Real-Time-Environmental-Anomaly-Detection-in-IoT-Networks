import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ==============================================
# 配置参数
# ==============================================
DATA_PATH = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete\preprocessed_time_series_augmented_100k.csv"
OUTPUT_DIR = r"D:\Oswaldo's surf project\DR O's database\final_preprocessed_data_complete"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 窗口参数（与原始代码保持一致）
WINDOW_SIZE = 24  # 24小时 = 1天（小时数据）
STRIDE = 1  # 窗口之间的步长（1 = 重叠窗口）
PREDICTION_HORIZON = 1  # 如果是预测；异常检测通常为0或1

# 用于异常检测的特征（使用增强后的标准化特征）
FEATURES = [
    'avg_PM2.5_normalized_scaled',
    'total_noise_duration_scaled',
    'noise_event_count_scaled',
    'avg_salience_scaled'
    # 可以根据需要添加 'avg_PM2.5'
]

# ==============================================
# 1. 加载并检查增强后的数据集
# ==============================================
print("=" * 60)
print("加载增强后的时间序列数据...")
print("=" * 60)

# 加载数据，第一列是时间戳
df = pd.read_csv(DATA_PATH)

# 重命名第一列为timestamp
time_column = df.columns[0]
df = df.rename(columns={time_column: 'timestamp'})

# 转换为datetime并设置为索引
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.set_index('timestamp')
df = df.sort_index()

print(f"数据形状: {df.shape}")
print(f"时间范围: {df.index.min()} → {df.index.max()}")
print(f"总样本数: {len(df)}")
print("列名:", df.columns.tolist())

# 检查缺失值
print("\n缺失值检查:")
print(df.isna().sum())

# 如果有缺失值，填充它们
if df.isna().sum().sum() > 0:
    print("发现缺失值，进行填充...")
    df = df.fillna(method='ffill').fillna(method='bfill')
    print("填充后缺失值:", df.isna().sum().sum())

# ==============================================
# 2. 检查所需特征是否存在
# ==============================================
print("\n" + "=" * 60)
print("检查特征列...")
print("=" * 60)

available_features = []
missing_features = []

for feature in FEATURES:
    if feature in df.columns:
        available_features.append(feature)
        print(f"✓ {feature}: 存在")
    else:
        missing_features.append(feature)
        print(f"✗ {feature}: 缺失")

# 如果有缺失的特征，尝试使用替代名称
if missing_features:
    print("\n尝试寻找替代特征...")
    for feature in missing_features[:]:  # 使用副本进行迭代
        if '_scaled' in feature:
            # 尝试去掉_scaled后缀
            alternative = feature.replace('_scaled', '')
            if alternative in df.columns:
                print(f"  使用 {alternative} 替代 {feature}")
                df[feature] = df[alternative]
                available_features.append(feature)
                missing_features.remove(feature)

if missing_features:
    print(f"警告: 以下特征仍然缺失: {missing_features}")
    print("将继续处理可用的特征...")

if not available_features:
    print("错误: 没有可用的特征列！")
    exit()

print(f"\n将使用以下特征创建窗口: {available_features}")

# ==============================================
# 3. 验证数据的统计特性
# ==============================================
print("\n" + "=" * 60)
print("数据统计特性")
print("=" * 60)

for feature in available_features:
    if feature in df.columns:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"{feature}:")
        print(f"  均值: {mean_val:.6f}, 标准差: {std_val:.6f}")
        print(f"  范围: [{min_val:.6f}, {max_val:.6f}]")

# ==============================================
# 4. 创建滑动窗口
# ==============================================
print("\n" + "=" * 60)
print(f"创建滑动窗口 (大小={WINDOW_SIZE}, 步长={STRIDE}, 预测视野={PREDICTION_HORIZON})")
print("=" * 60)


def create_sliding_windows(data, window_size, stride=1, horizon=0):
    """
    创建3D滑动窗口数组: (n_samples, window_size, n_features)
    如果 horizon > 0 → 同时返回预测目标值
    """
    X = []
    y = [] if horizon > 0 else None

    n_samples = len(data)

    for i in range(0, n_samples - window_size - horizon + 1, stride):
        window = data[i: i + window_size]
        X.append(window)

        if horizon > 0:
            target = data[i + window_size: i + window_size + horizon]
            y.append(target)

    X = np.array(X)
    if y is not None:
        y = np.array(y)

    return X, y


# 使用标准化特征创建滑动窗口
data_array = df[available_features].values

X_windows, y_windows = create_sliding_windows(
    data_array,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    horizon=PREDICTION_HORIZON
)

print(f"窗口形状: {X_windows.shape}")  # (n_windows, timesteps, n_features)
print(f"窗口数量: {X_windows.shape[0]:,}")
print(f"每个窗口的时间步: {X_windows.shape[1]}")
print(f"每个时间步的特征数: {X_windows.shape[2]}")

if y_windows is not None:
    print(f"目标形状: {y_windows.shape}")

# ==============================================
# 5. 窗口数据质量检查
# ==============================================
print("\n" + "=" * 60)
print("窗口数据质量检查")
print("=" * 60)

# 检查NaN值
nan_count = np.isnan(X_windows).sum()
print(f"NaN值数量: {nan_count}")

if nan_count > 0:
    print("发现NaN值，进行填充...")
    # 使用前向填充，然后后向填充
    for i in range(X_windows.shape[0]):
        for j in range(X_windows.shape[2]):
            col_data = X_windows[i, :, j]
            if np.isnan(col_data).any():
                df_series = pd.Series(col_data)
                df_series = df_series.fillna(method='ffill').fillna(method='bfill')
                X_windows[i, :, j] = df_series.values

# 检查无限值
inf_count = np.isinf(X_windows).sum()
print(f"无限值数量: {inf_count}")

# 窗口统计
print(f"\n窗口数据统计:")
print(f"  均值: {X_windows.mean():.6f}")
print(f"  标准差: {X_windows.std():.6f}")
print(f"  最小值: {X_windows.min():.6f}")
print(f"  最大值: {X_windows.max():.6f}")

# ==============================================
# 6. 保存处理后的数据
# ==============================================
print("\n" + "=" * 60)
print("保存处理后的数据")
print("=" * 60)

# 保存滑动窗口
window_path = os.path.join(OUTPUT_DIR, "x_windows_100k.npy")
np.save(window_path, X_windows)
print(f"✓ 滑动窗口保存到: {window_path}")
print(f"  文件大小: {os.path.getsize(window_path) / (1024 * 1024):.2f} MB")

# 保存目标值（如果有）
if y_windows is not None:
    y_path = os.path.join(OUTPUT_DIR, "y_windows_100k.npy")
    np.save(y_path, y_windows)
    print(f"✓ 目标值保存到: {y_path}")

# 保存处理后的数据框（包含时间戳）
df_processed = df.copy()
processed_csv_path = os.path.join(OUTPUT_DIR, "processed_time_series_augmented_100k.csv")
df_processed.reset_index().to_csv(processed_csv_path, index=False)
print(f"✓ 处理后的时间序列保存到: {processed_csv_path}")

# ==============================================
# 7. 生成数据报告
# ==============================================
print("\n" + "=" * 60)
print("生成数据报告")
print("=" * 60)

# 创建报告
report = {
    'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'input_file': DATA_PATH,
    'output_files': {
        'x_windows': window_path,
        'processed_time_series': processed_csv_path
    },
    'data_statistics': {
        'original_samples': len(df),
        'window_count': X_windows.shape[0],
        'window_size': X_windows.shape[1],
        'features_per_timestep': X_windows.shape[2],
        'time_range': [str(df.index.min()), str(df.index.max())]
    },
    'window_settings': {
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'prediction_horizon': PREDICTION_HORIZON
    },
    'features_used': available_features,
    'data_quality': {
        'nan_values_before_filling': int(nan_count),
        'inf_values': int(inf_count),
        'window_data_mean': float(X_windows.mean()),
        'window_data_std': float(X_windows.std())
    }
}

# 添加特征统计
feature_stats = {}
for feature in available_features:
    if feature in df.columns:
        feature_stats[feature] = {
            'mean': float(df[feature].mean()),
            'std': float(df[feature].std()),
            'min': float(df[feature].min()),
            'max': float(df[feature].max())
        }
report['feature_statistics'] = feature_stats

# 保存报告
report_path = os.path.join(OUTPUT_DIR, "100k_window_processing_report.json")
import json

with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"✓ 处理报告保存到: {report_path}")

# ==============================================
# 8. 可视化（可选）
# ==============================================
print("\n" + "=" * 60)
print("生成可视化图表")
print("=" * 60)

try:
    # 创建可视化
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # 1. 时间序列图（前500个点）
    for i, feature in enumerate(available_features[:4]):
        ax = axes[i // 2, i % 2]
        plot_data = df[feature].iloc[:500]
        ax.plot(plot_data.index, plot_data.values, 'b-', alpha=0.7, linewidth=1)
        ax.set_title(f'{feature} (前500个点)')
        ax.set_xlabel('时间')
        ax.set_ylabel('标准化值')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    # 2. 第一个窗口可视化
    if X_windows.shape[0] > 0:
        ax = axes[2, 0]
        for i in range(min(4, X_windows.shape[2])):
            ax.plot(X_windows[0, :, i], label=f'特征 {i + 1}')
        ax.set_title(f'第一个滑动窗口 (大小={WINDOW_SIZE})')
        ax.set_xlabel('时间步')
        ax.set_ylabel('标准化值')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. 窗口数量统计
    ax = axes[2, 1]
    n_windows = X_windows.shape[0]
    ax.bar(['总窗口数'], [n_windows], color='skyblue')
    ax.set_title(f'滑动窗口总数: {n_windows:,}')
    ax.set_ylabel('数量')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    viz_path = os.path.join(OUTPUT_DIR, "100k_window_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 可视化图表保存到: {viz_path}")

except Exception as e:
    print(f"⚠ 可视化生成失败: {e}")

# ==============================================
# 9. 验证保存的数据
# ==============================================
print("\n" + "=" * 60)
print("验证保存的数据")
print("=" * 60)

# 重新加载保存的窗口数据
try:
    X_loaded = np.load(window_path)
    print(f"✓ 成功加载 x_windows_100k.npy")
    print(f"  形状: {X_loaded.shape}")
    print(f"  数据类型: {X_loaded.dtype}")

    # 验证数据完整性
    loaded_nan = np.isnan(X_loaded).sum()
    loaded_inf = np.isinf(X_loaded).sum()
    print(f"  NaN值: {loaded_nan}")
    print(f"  无限值: {loaded_inf}")

    if loaded_nan == 0 and loaded_inf == 0:
        print("  ✅ 数据完整，无异常值")
    else:
        print("  ⚠ 发现异常值，可能需要重新处理")

except Exception as e:
    print(f"✗ 加载验证失败: {e}")

# ==============================================
# 10. 最终总结
# ==============================================
print("\n" + "=" * 60)
print("🎉 数据处理完成!")
print("=" * 60)

print(f"\n📊 处理结果:")
print(f"  输入数据: {len(df)} 个样本")
print(f"  创建窗口: {X_windows.shape[0]:,} 个")
print(f"  每个窗口: {X_windows.shape[1]} 时间步, {X_windows.shape[2]} 特征")
print(f"  时间范围: {df.index.min()} 到 {df.index.max()}")

print(f"\n📁 生成的文件:")
print(f"  1. x_windows_100k.npy - 滑动窗口数据 ({X_windows.shape[0]:,} 窗口)")
print(f"  2. processed_time_series_augmented.csv - 处理后的时间序列")
print(f"  3. window_processing_report.json - 处理报告")
print(f"  4. window_visualization.png - 可视化图表")

print(f"\n🔧 下一步:")
print(f"  现在可以使用 x_windows_100k.npy 进行模型训练了!")
print(f"  示例加载代码:")
print(f"  X_windows = np.load(r\"{window_path}\")")
print(f"  print(f\"窗口形状: {{X_windows.shape}}\")")
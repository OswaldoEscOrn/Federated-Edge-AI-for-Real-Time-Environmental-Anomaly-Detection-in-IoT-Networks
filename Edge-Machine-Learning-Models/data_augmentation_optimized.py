import pandas as pd
import numpy as np
import os


def augment_to_100k_with_continuity_fixed(original_path, output_path, target_samples=100000):
    """
    在标准化空间内进行数据增强（保持z-score标准化特性）
    """
    print("🚀 开始数据增强（z-score标准化空间内）...")

    # 1. 加载原始数据
    df = pd.read_csv(original_path)
    print(f"📊 原始数据形状: {df.shape}")
    print(f"📋 列名: {df.columns.tolist()}")

    # 2. 检查所需列
    scaled_features = ['avg_PM2.5_normalized_scaled', 'total_noise_duration_scaled',
                       'noise_event_count_scaled', 'avg_salience_scaled']

    for col in scaled_features:
        if col not in df.columns:
            raise ValueError(f"❌ 缺少标准化列: {col}")

    if 'avg_PM2.5' not in df.columns:
        raise ValueError(f"❌ 缺少原始PM2.5列: avg_PM2.5")

    # 3. 提取标准化特征（z-score标准化，不是0-1标准化）
    X_scaled = df[scaled_features].values
    pm25_original = df['avg_PM2.5'].values

    print(f"📊 标准化特征范围验证（z-score标准化）:")
    for i, col in enumerate(scaled_features):
        min_val = X_scaled[:, i].min()
        max_val = X_scaled[:, i].max()
        mean_val = X_scaled[:, i].mean()
        std_val = X_scaled[:, i].std()
        print(f"  {col}: 均值={mean_val:.3f}, 标准差={std_val:.3f}, 范围=[{min_val:.3f}, {max_val:.3f}]")

    # 4. 生成目标数量的数据点
    original_len = len(df)
    repeats_needed = int(np.ceil(target_samples / original_len)) + 2

    print(f"🔄 原始长度: {original_len}, 目标长度: {target_samples}")
    print(f"🔄 需要重复次数: {repeats_needed}")

    # 存储所有增强数据
    all_augmented_scaled = []
    all_augmented_pm25 = []
    all_timestamps = []

    # 起始时间
    current_time = pd.Timestamp('2013-03-01 00:00:00')

    # 不同的季节噪声模式
    season_patterns = [
        {'name': 'spring', 'base_noise': 0.1, 'trend_factor': 0.5},  # 调整噪声水平
        {'name': 'summer', 'base_noise': 0.15, 'trend_factor': 0.8},
        {'name': 'fall', 'base_noise': 0.08, 'trend_factor': 0.3},
        {'name': 'winter', 'base_noise': 0.2, 'trend_factor': 1.0}
    ]

    for i in range(repeats_needed):
        current_samples = len(np.concatenate(all_augmented_scaled)) if all_augmented_scaled else 0
        if current_samples >= target_samples:
            break

        # 获取当前季节模式
        season = season_patterns[i % len(season_patterns)]

        # 复制原始标准化数据
        chunk_scaled = X_scaled.copy()

        # 在z-score标准化空间内添加噪声（不需要裁剪到0-1）
        noise = np.random.normal(0, season['base_noise'], chunk_scaled.shape)

        # 添加轻微的时间趋势（模拟日变化）
        time_trend = np.linspace(-0.1, 0.1, len(chunk_scaled)).reshape(-1, 1)
        time_trend = np.tile(time_trend, (1, chunk_scaled.shape[1]))

        noise += time_trend * season['trend_factor']

        # 应用噪声（保持z-score标准化特性，不裁剪）
        chunk_augmented = chunk_scaled + noise

        # 对于PM2.5原始值，我们基于增强后的标准化值进行轻微调整
        pm25_season_factor = 1.0 + 0.1 * np.sin(i * np.pi / 2)  # 季节性因子
        pm25_chunk = pm25_original * pm25_season_factor

        # 添加一些随机性
        pm25_noise = np.random.normal(0, pm25_original.std() * 0.1, len(pm25_chunk))
        pm25_chunk = np.clip(pm25_chunk + pm25_noise, 0, None)  # PM2.5不能为负

        # 生成时间戳（注意：频率是小写 'h'，不是大写 'H'）
        chunk_timestamps = pd.date_range(
            start=current_time,
            periods=len(chunk_augmented),
            freq='h'  # 修正这里：小写 'h'
        )
        current_time = chunk_timestamps[-1] + pd.Timedelta(hours=1)

        # 添加到结果列表
        all_augmented_scaled.append(chunk_augmented)
        all_augmented_pm25.append(pm25_chunk)
        all_timestamps.extend(chunk_timestamps)

        print(f"  🔄 完成第 {i + 1} 个区块，季节: {season['name']}, 当前总样本: {current_samples + len(chunk_augmented)}")

    # 5. 合并所有数据
    print("🔗 合并数据块...")
    final_scaled = np.vstack(all_augmented_scaled)[:target_samples]
    final_pm25 = np.hstack(all_augmented_pm25)[:target_samples]
    final_timestamps = all_timestamps[:target_samples]

    # 6. 创建最终DataFrame
    augmented_df = pd.DataFrame(
        final_scaled,
        columns=scaled_features,
        index=final_timestamps
    )
    augmented_df['avg_PM2.5'] = final_pm25

    print(f"✅ 增强完成！最终形状: {augmented_df.shape}")
    print(f"⏰ 时间范围: {augmented_df.index[0]} 到 {augmented_df.index[-1]}")

    # 7. 验证增强结果
    print("\n📊 增强数据验证:")
    print("标准化特征统计（z-score标准化）:")
    for col in scaled_features:
        min_val = augmented_df[col].min()
        max_val = augmented_df[col].max()
        mean_val = augmented_df[col].mean()
        std_val = augmented_df[col].std()
        print(f"  {col}: 均值={mean_val:.3f}, 标准差={std_val:.3f}, 范围=[{min_val:.3f}, {max_val:.3f}]")

    print(f"\n原始PM2.5统计:")
    print(f"  最小值: {augmented_df['avg_PM2.5'].min():.2f}")
    print(f"  最大值: {augmented_df['avg_PM2.5'].max():.2f}")
    print(f"  均值: {augmented_df['avg_PM2.5'].mean():.2f}")
    print(f"  标准差: {augmented_df['avg_PM2.5'].std():.2f}")
    print(f"  负值数量: {(augmented_df['avg_PM2.5'] < 0).sum()}")

    # 8. 保存数据
    augmented_df.to_csv(output_path)
    print(f"💾 数据已保存到: {output_path}")

    return augmented_df


def create_sliding_windows_simple(data, window_size=24, stride=1):
    """
    创建滑动窗口（简化版）
    """
    print(f"\n🎯 创建滑动窗口（窗口大小={window_size}, 步长={stride})...")

    # 只使用标准化特征
    scaled_features = ['avg_PM2.5_normalized_scaled', 'total_noise_duration_scaled',
                       'noise_event_count_scaled', 'avg_salience_scaled']

    X = data[scaled_features].values

    n_samples = len(X)
    n_features = X.shape[1]

    # 计算窗口数量
    n_windows = (n_samples - window_size) // stride + 1

    # 创建窗口
    windows = np.zeros((n_windows, window_size, n_features))

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = X[start:end]

    print(f"  原始样本数: {n_samples}")
    print(f"  特征数量: {n_features}")
    print(f"  创建的窗口数: {n_windows}")
    print(f"  窗口形状: {windows.shape}")

    return windows


def create_and_save_training_data(augmented_df, output_dir):
    """
    创建并保存训练数据
    """
    print("\n📁 创建训练数据集...")

    # 1. 创建滑动窗口
    X_windows_100k = create_sliding_windows_simple(augmented_df, window_size=24, stride=1)

    # 2. 分割数据集（85%训练，15%验证）
    split_idx = int(len(X_windows_100k) * 0.85)
    X_train = X_windows_100k[:split_idx]
    X_val = X_windows_100k[split_idx:]

    print(f"\n📊 数据集分割:")
    print(f"  训练集: {X_train.shape[0]} 个窗口 ({split_idx / len(X_windows_100k) * 100:.1f}%)")
    print(f"  验证集: {X_val.shape[0]} 个窗口 ({len(X_val) / len(X_windows_100k) * 100:.1f}%)")

    # 3. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 4. 保存数据
    np.save(os.path.join(output_dir, "X_windows_100k.npy"), X_windows_100k)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)

    print(f"\n💾 文件已保存到: {output_dir}")
    print(f"  - X_windows_100k.npy: 形状 {X_windows_100k.shape}")
    print(f"  - X_train.npy: 形状 {X_train.shape}")
    print(f"  - X_val.npy: 形状 {X_val.shape}")

    return X_windows_100k, X_train, X_val


# ====================== 主程序 ======================
if __name__ == "__main__":
    # 配置文件路径
    ORIGINAL_DATA_PATH = r"D:\Oswaldo's surf project\DR O's database\preprocessed_data\normalized_hourly_data.csv"
    OUTPUT_DIR = r"D:\Oswaldo's surf project\DR O's database\preprocessed_data"
    AUGMENTED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "augmented_100k_fixed.csv")

    print("=" * 60)
    print("🚀 100k数据增强系统（z-score标准化空间内）")
    print("=" * 60)

    try:
        # 第一步：生成10万个增强数据点
        print("\n📦 第一步：生成增强数据")
        augmented_data = augment_to_100k_with_continuity_fixed(
            original_path=ORIGINAL_DATA_PATH,  # 明确传递参数
            output_path=AUGMENTED_OUTPUT_PATH,
            target_samples=100000
        )

        # 第二步：创建训练窗口
        print("\n🎯 第二步：创建训练窗口")
        X_windows_100k, X_train, X_val = create_and_save_training_data(
            augmented_df=augmented_data,  # 明确传递参数
            output_dir=OUTPUT_DIR
        )

        # 第三步：数据验证
        print("\n🔍 第三步：数据验证")
        print("窗口数据统计:")
        print(f"  X_windows_100k 形状: {X_windows_100k.shape}")
        print(f"  X_windows_100k 均值: {X_windows_100k.mean():.4f}")
        print(f"  X_windows_100k 标准差: {X_windows_100k.std():.4f}")
        print(f"  X_windows_100k 范围: [{X_windows_100k.min():.4f}, {X_windows_100k.max():.4f}]")

        print(f"\n数据完整性检查:")
        print(f"  NaN值数量: {np.isnan(X_windows_100k).sum()}")
        print(f"  无限值数量: {np.isinf(X_windows_100k).sum()}")

        # 第四步：使用指南
        print("\n" + "=" * 60)
        print("📋 使用指南")
        print("=" * 60)
        print("✅ 数据增强完成！")
        print("\n📁 生成的文件:")
        print(f"  1. {AUGMENTED_OUTPUT_PATH} - 增强后的完整数据集")
        print(f"  2. {OUTPUT_DIR}/X_windows_100k.npy - 滑动窗口数据")
        print(f"  3. {OUTPUT_DIR}/X_train.npy - 训练集")
        print(f"  4. {OUTPUT_DIR}/X_val.npy - 验证集")

        print("\n🔧 如何使用:")
        print("1. 加载数据:")
        print("   X_windows_100k = np.load('X_windows_100k.npy')")
        print("2. 训练AutoEncoder:")
        print("   model.fit(X_train, X_train, validation_data=(X_val, X_val))")
        print("3. 特征使用了z-score标准化（均值约0，标准差约1）")
        print("4. 保持了时间序列连续性")

        print("\n📊 数据统计:")
        print(f"  总样本数: {len(augmented_data)}")
        print(f"  总窗口数: {len(X_windows_100k)}")
        print(f"  训练窗口: {len(X_train)} ({len(X_train) / len(X_windows_100k) * 100:.1f}%)")
        print(f"  验证窗口: {len(X_val)} ({len(X_val) / len(X_windows_100k) * 100:.1f}%)")

    except Exception as e:
        print(f"❌ 错误发生: {e}")
        import traceback

        traceback.print_exc()
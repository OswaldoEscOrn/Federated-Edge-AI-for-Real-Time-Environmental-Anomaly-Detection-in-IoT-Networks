import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


# ====================== 核心函数 - 分层模式提取（修改为4个特征） ======================
def extract_hierarchical_patterns(df_aircraft, df_pm25):
    """
    分层提取模式特征（全局 + 24小时级 + 时段级） - 只提取4个核心特征
    :param df_aircraft: 已预处理的飞机噪音数据
    :param df_pm25: 已预处理的PM2.5数据
    :return: 分层模式字典
    """
    patterns = {}
    pm25_city_cols = [col for col in df_pm25.columns if col != 'timestamp']  # 所有城市PM2.5列

    # 第一层：全局统计特征（精简为4个核心）
    patterns['global'] = {
        'total_aircraft_records': len(df_aircraft),
        'overall_noise_mean': df_aircraft['max_slow'].mean() if len(df_aircraft) > 0 else 0,
        'overall_pm25_mean': df_pm25[pm25_city_cols].values.flatten().mean() if len(df_pm25) > 0 else 0,
        'aircraft_type_diversity': df_aircraft['type'].nunique() / len(df_aircraft) if len(df_aircraft) > 0 else 0
    }

    # 第二层：24小时级模式特征（只提取4个核心特征）
    for hour in range(24):
        aircraft_hour = df_aircraft[df_aircraft['timestamp'].dt.hour == hour]
        pm25_hour = df_pm25[df_pm25['timestamp'].dt.hour == hour]
        hour_features = {}

        # 只提取4个核心特征（与真实数据对齐）
        # 1. PM2.5特征 (对应真实数据的 avg_PM2.5)
        if len(pm25_hour) > 0:
            pm25_hour_values = pm25_hour[pm25_city_cols].values.flatten()
            hour_features['pm25_mean'] = pm25_hour_values.mean()
        else:
            hour_features['pm25_mean'] = 0

        # 2. 飞机噪音特征 (对应真实数据的 total_noise_duration)
        hour_features['noise_mean'] = aircraft_hour['max_slow'].mean() if len(aircraft_hour) > 0 else 0

        # 3. 飞机数量特征 (对应真实数据的 noise_event_count)
        hour_features['aircraft_count'] = len(aircraft_hour)

        # 4. 飞机类型多样性特征 (对应真实数据的 avg_salience)
        hour_features['aircraft_type_diversity'] = aircraft_hour['type'].nunique() / len(aircraft_hour) if len(
            aircraft_hour) > 0 else 0

        patterns[f'hour_{hour}'] = hour_features

    # 第三层：时段级模式特征（只提取4个核心特征）
    time_windows = {'morning': (6, 10), 'daytime': (10, 17), 'evening': (17, 21), 'night': (21, 6)}
    for period, (start, end) in time_windows.items():
        period_features = {}

        # 飞机时段筛选
        if start < end:
            aircraft_mask = (df_aircraft['timestamp'].dt.hour >= start) & (df_aircraft['timestamp'].dt.hour < end)
        else:
            aircraft_mask = (df_aircraft['timestamp'].dt.hour >= start) | (df_aircraft['timestamp'].dt.hour < end)
        aircraft_period = df_aircraft[aircraft_mask]

        # PM2.5时段筛选
        if start < end:
            pm25_mask = (df_pm25['timestamp'].dt.hour >= start) & (df_pm25['timestamp'].dt.hour < end)
        else:
            pm25_mask = (df_pm25['timestamp'].dt.hour >= start) | (df_pm25['timestamp'].dt.hour < end)
        pm25_period = df_pm25[pm25_mask]

        # 只提取4个核心特征
        period_features['aircraft_count'] = len(aircraft_period)
        period_features['noise_mean'] = aircraft_period['max_slow'].mean() if len(aircraft_period) > 0 else 0
        period_features['pm25_mean'] = pm25_period[pm25_city_cols].values.flatten().mean() if len(
            pm25_period) > 0 else 0
        period_features['aircraft_type_diversity'] = aircraft_period['type'].nunique() / len(aircraft_period) if len(
            aircraft_period) > 0 else 0

        patterns[f'period_{period}'] = period_features

    print("✅ 分层模式特征提取完成（4个核心特征：PM2.5均值、噪音均值、飞机数量、飞机类型多样性）")
    return patterns


# ====================== 模式特征转DAE输入（修改为4个特征） ======================
def patterns_to_dae_features(patterns_dict):
    """转换为DeepAutoEncoder可用的标准化特征（4个特征版本）"""
    # 4个核心特征
    hour_feature_names = ['pm25_mean', 'noise_mean', 'aircraft_count', 'aircraft_type_diversity']

    # 24小时特征矩阵（核心） - 每个小时4个特征
    hour_matrix = np.array([[patterns_dict[f'hour_{h}'][feat] for feat in hour_feature_names] for h in range(24)])

    # 全局+时段特征向量（精简版）
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

    # 标准化 - 分开标准化小时特征和全局特征
    hour_scaler = MinMaxScaler(feature_range=(0, 1))
    global_scaler = MinMaxScaler(feature_range=(0, 1))

    hour_matrix_scaled = hour_scaler.fit_transform(hour_matrix)
    global_period_scaled = global_scaler.fit_transform(global_period_vector)

    # 特征名称映射
    feature_mapping = {
        'hour_feature_names': hour_feature_names,
        'hour_feature_descriptions': {
            'pm25_mean': 'PM2.5平均值（对应真实数据的avg_PM2.5）',
            'noise_mean': '噪音平均值（对应真实数据的total_noise_duration）',
            'aircraft_count': '飞机数量（对应真实数据的noise_event_count）',
            'aircraft_type_diversity': '飞机类型多样性（对应真实数据的avg_salience）'
        },
        'global_feature_names': ['total_aircraft_records', 'overall_noise_mean', 'overall_pm25_mean',
                                 'aircraft_type_diversity'],
        'period_feature_names': [f'period_{p}_{f}' for p in ['morning', 'daytime', 'evening', 'night']
                                 for f in ['aircraft_count', 'noise_mean', 'pm25_mean', 'aircraft_type_diversity']],
        'real_data_features': ['avg_PM2.5', 'total_noise_duration', 'noise_event_count', 'avg_salience']
    }

    print("✅ 4特征转换完成：")
    print(f"  - 24小时特征矩阵形状：{hour_matrix_scaled.shape} (24小时 × 4特征)")
    print(f"  - 全局+时段特征向量形状：{global_period_scaled.shape}")
    print(f"  - 特征对应关系：")
    print(f"     虚拟数据: {hour_feature_names}")
    print(f"     真实数据: {feature_mapping['real_data_features']}")

    return hour_matrix_scaled, global_period_scaled, feature_mapping


# ====================== 生成4特征虚拟数据样本（新功能） ======================
def generate_4feature_virtual_samples(hour_matrix_scaled, n_samples=35040, variation_factor=0.1):
    """
    基于提取的模式生成4特征的虚拟数据样本

    参数:
        hour_matrix_scaled: 标准化的小时特征矩阵 (24, 4)
        n_samples: 生成的样本数量
        variation_factor: 变异因子，控制样本的多样性
    """
    print(f"\n🔄 生成4特征的虚拟数据样本 (n={n_samples})...")

    # 基础模式
    base_pattern = hour_matrix_scaled  # (24, 4)

    # 生成变异样本
    virtual_samples = []
    for i in range(n_samples):
        # 为每个样本添加随机变异
        variation = np.random.normal(0, variation_factor, base_pattern.shape)
        sample = base_pattern + variation

        # 确保值在合理范围内
        sample = np.clip(sample, 0, 1)
        virtual_samples.append(sample)

    virtual_samples = np.array(virtual_samples)  # (n_samples, 24, 4)
    print(f"  生成的虚拟数据形状: {virtual_samples.shape}")

    return virtual_samples


# ====================== 主流程（直接读取预处理后的数据） ======================
if __name__ == "__main__":
    # ---------------------- 关键：读取你已预处理好的文件 ----------------------
    # 请确认以下路径是你预处理后文件的实际路径，修改后运行！
    processed_pm25_path = r"D:\Oswaldo's surf project\My Database\PM2.5\random_city3PM25_data.csv"
    processed_noise_path = r"D:\Oswaldo's surf project\My Database\aircraft noise_data_test4.csv"

    # 读取数据（已预处理，直接加载）
    pm25_df = pd.read_csv(processed_pm25_path)
    noise_df = pd.read_csv(processed_noise_path)

    # 转换时间列格式（确保dt.hour可用，必须保留）
    pm25_df['timestamp'] = pd.to_datetime(pm25_df['timestamp'])
    noise_df['timestamp'] = pd.to_datetime(noise_df['timestamp'])

    print(f"✅ 已加载预处理后的数据：")
    print(f"  - PM2.5数据：{pm25_df.shape} 条记录，缺失值：{pm25_df.isnull().sum().sum()} 个")
    print(f"  - 飞机噪音数据：{noise_df.shape} 条记录，缺失值：{noise_df.isnull().sum().sum()} 个")

    # 提取模式特征（4个特征版本）
    hierarchical_patterns = extract_hierarchical_patterns(noise_df, pm25_df)

    # 转换为DAE特征并保存（4个特征版本）
    hour_matrix_scaled, global_period_scaled, feature_mapping = patterns_to_dae_features(hierarchical_patterns)

    # 保存结果（路径可修改）
    save_path = r"D:\Oswaldo's surf project\My Database\Merge_and_DAE_4_extracted_features"
    import os

    os.makedirs(save_path, exist_ok=True)

    # 保存4特征版本的文件
    np.save(os.path.join(save_path, "24hour_pattern_matrix_4features_scaled.npy"), hour_matrix_scaled)
    np.save(os.path.join(save_path, "global_period_pattern_vector_4features_scaled.npy"), global_period_scaled)

    # 生成并保存4特征的虚拟数据样本
    virtual_samples = generate_4feature_virtual_samples(hour_matrix_scaled, n_samples=35040)
    np.save(os.path.join(save_path, "virtual_samples_4features_35040.npy"), virtual_samples)

    # 保存特征映射
    import json

    with open(os.path.join(save_path, "feature_mapping_4features.json"), 'w', encoding='utf-8') as f:
        json.dump(feature_mapping, f, ensure_ascii=False, indent=2)

    # 打印验证信息
    print("\n📊 示例：0点模式特征（4个特征，标准化后）：")
    print(f"  - PM2.5均值: {hour_matrix_scaled[0][0]:.3f}")
    print(f"  - 噪音均值: {hour_matrix_scaled[0][1]:.3f}")
    print(f"  - 飞机数量: {hour_matrix_scaled[0][2]:.3f}")
    print(f"  - 飞机类型多样性: {hour_matrix_scaled[0][3]:.3f}")

    # 特征统计信息
    print("\n📈 4特征统计信息：")
    for i, feat_name in enumerate(feature_mapping['hour_feature_names']):
        feat_data = hour_matrix_scaled[:, i]
        print(
            f"  {feat_name}: 均值={feat_data.mean():.3f}, 标准差={feat_data.std():.3f}, 范围=[{feat_data.min():.3f}, {feat_data.max():.3f}]")

    # 虚拟数据质量检查
    print("\n🔍 虚拟数据质量检查：")
    print(f"  虚拟数据形状: {virtual_samples.shape}")
    print(f"  虚拟数据范围: [{virtual_samples.min():.3f}, {virtual_samples.max():.3f}]")
    print(f"  虚拟数据均值: {virtual_samples.mean():.3f} ± {virtual_samples.std():.3f}")
    print(f"  缺失值数量: {np.isnan(virtual_samples).sum()}")

    print(f"\n🎉 4特征版本运行完成！已保存文件到: {save_path}")
    print("📁 生成的文件：")
    print(f"  1. 24hour_pattern_matrix_4features_scaled.npy - 24小时4特征模式矩阵")
    print(f"  2. global_period_pattern_vector_4features_scaled.npy - 全局+时段特征向量")
    print(f"  3. virtual_samples_4features_35040.npy - 4特征虚拟数据样本 (35040个)")
    print(f"  4. feature_mapping_4features.json - 特征映射文件")
    print("\n💡 现在可以直接使用这些4特征文件进行后续的混合数据集创建！")
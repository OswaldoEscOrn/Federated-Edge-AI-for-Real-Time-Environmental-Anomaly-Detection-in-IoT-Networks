import pandas as pd

#
input_path = r"D:\Oswaldo's surf project\My Database\aircraft noise_data_test2.csv"
output_path = r"D:\Oswaldo's surf project\My Database\aircraft noise_data_test4.csv"

# 1. 读取原始数据
df = pd.read_csv(input_path)
print("原始数据形状:", df.shape)
print("原始缺失值统计:")
print(df.isnull().sum())

# 2. 确保 max_slow 列为数值类型（非数值转为 NaN）
df['max_slow'] = pd.to_numeric(df['max_slow'], errors='coerce')

# 3. 处理缺失值
# 3.1 timestamp 列：前向填充
df['timestamp'] = df['timestamp'].fillna(method='ffill')

# 3.2 type 和 model 列：众数填充
type_mode = df['type'].mode()[0] if not df['type'].mode().empty else None
model_mode = df['model'].mode()[0] if not df['model'].mode().empty else None
df['type'] = df['type'].fillna(type_mode)
df['model'] = df['model'].fillna(model_mode)

# 4. 输出处理后的统计信息
print("\n处理后的缺失值统计:")
print(df.isnull().sum())
print(f"\n使用的众数填充值:")
print(f"type 列众数: {type_mode}")
print(f"model 列众数: {model_mode}")

# 5. 保存最终文件
df.to_csv(output_path, index=False)
print(f"\n处理后的数据已保存到: {output_path}")
print(f"最终数据形状: {df.shape}")
print("\n数据前5行预览:")
print(df.head())
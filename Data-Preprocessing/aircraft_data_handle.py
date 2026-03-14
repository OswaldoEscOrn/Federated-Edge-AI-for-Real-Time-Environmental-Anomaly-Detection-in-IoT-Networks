import pandas as pd


input_path = r"D:\Oswaldo's surf project\My Database\aircraft noise_data_test2.csv"
output_path = r"D:\Oswaldo's surf project\My Database\aircraft noise_data_test4.csv"

# 1. Load original data
df = pd.read_csv(input_path)
print("Original data shape:", df.shape)
print("Original missing value statistics:")
print(df.isnull().sum())

# 2. Ensure max_slow column is numeric (convert non-numeric to NaN)
df['max_slow'] = pd.to_numeric(df['max_slow'], errors='coerce')

# 3. Handle missing values
# 3.1 timestamp column: forward fill
df['timestamp'] = df['timestamp'].fillna(method='ffill')

# 3.2 type and model columns: mode fill
type_mode = df['type'].mode()[0] if not df['type'].mode().empty else None
model_mode = df['model'].mode()[0] if not df['model'].mode().empty else None
df['type'] = df['type'].fillna(type_mode)
df['model'] = df['model'].fillna(model_mode)

# 4. Output statistics after processing
print("\nMissing value statistics (after processing):")
print(df.isnull().sum())
print(f"\nMode values used for filling:")
print(f"type column mode: {type_mode}")
print(f"model column mode: {model_mode}")

# 5. Save final file
df.to_csv(output_path, index=False)
print(f"\nProcessed data saved to: {output_path}")
print(f"Final data shape: {df.shape}")
print("\nPreview of first 5 rows:")
print(df.head())

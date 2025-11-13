import pandas as pd
import os

# 定义输入文件
input_file = 'Gen_data/AGGREGATED_GENERATION_PER_TYPE_GENERATION_15min_2025.csv'
# 定义输出文件夹
output_dir = 'generation_by_type_2025'

# 假定文件结构和之前的代码一致：
# Col 0: Time Interval
# Col 1: Country/Area
# Col 2: Generation Type (这是我们用于拆分的列)
# Col 3: Generation Value
COLUMNS = ['MTU (CET/CEST)', 'Area', 'Production Type', 'Generation(MW)']

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建输出目录: {output_dir}")

try:
    # Step 1: 读取 CSV 文件
    # 使用 header=None 读取，并指定列名，避免因文件开头不一致导致的错误。
    # 假设第一行是标题，我们跳过它。
    df = pd.read_csv(
        input_file,
        header=None,
        names=COLUMNS,
        skiprows=1  # 假设数据有标题行，跳过它
    )

    # 再次检查并确保 'Generation_Value' 列是数值类型 (可选但推荐，以防数据中仍有脏数据)
    df['Generation(MW)'] = pd.to_numeric(df['Generation(MW)'], errors='coerce')

    # Step 2: 获取所有不同的 Generation Type
    production_types = df['Production Type'].unique()

    # 记录处理类型数量
    count = 0

    print(f"文件中发现 {len(production_types)} 种不同的发电类型，开始拆分...")

    # Step 3 & 4: 遍历所有类型，并保存为新的 CSV 文件
    for gen_type in production_types:
        if pd.isna(gen_type):
            # 跳过空值（NaN）类型
            print("⚠️ 跳过空/缺失值的发电类型。")
            continue

        # 1. 清理类型名称，用于安全的文件名 (替换特殊字符为空格，然后用下划线连接)
        # 例如 'Hydro Pumped Storage' -> 'Hydro_Pumped_Storage'
        safe_type_name = str(gen_type).replace('(', '').replace(')', '').replace('/', '_').replace(' ', '_').strip()

        # 2. 构造输出文件名
        output_filename = os.path.join(output_dir, f"{safe_type_name}_15min.csv")

        # 3. 过滤数据
        df_filtered = df[df['Production Type'] == gen_type].copy()

        # 4. 保存文件 (不包含索引)
        df_filtered.to_csv(output_filename, index=False, encoding='utf-8')

        count += 1
        print(f"   - 已处理并保存: {gen_type} -> {output_filename}")

    print("-" * 50)
    print(f"✅ 批处理完成。总共生成了 {count} 个文件，保存在目录 '{output_dir}' 中。")
    print("-" * 50)

except FileNotFoundError:
    print(f"错误：文件未找到。请确保文件名 '{input_file}' 正确且文件存在。")
except Exception as e:
    print(f"处理过程中发生未知错误: {e}")
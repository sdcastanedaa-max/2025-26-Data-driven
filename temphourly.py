import pandas as pd

# 定义输入和输出文件名
# 请注意：这里假设您新文件的名称是 'TotalGen.csv'
input_file = 'generation_by_type/Wind_Onshore_GENERATION.csv'
output_file = 'generation_by_type/Wind_Onshore_Hourly.csv'

# 定义正确的时间格式字符串 (日/月/年 时:分:秒)
DATE_FORMAT = "%d/%m/%Y %H:%M:%S"

try:
    # Step 1: 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 检查所需的列是否存在，现在需要 'Generation_Type'
    required_cols = ['Time_Interval', 'Country_Area', 'Production_Type', 'Total_Generation']
    if not all(col in df.columns for col in required_cols):
        print(f"错误：输入文件缺少必需的列: {required_cols}")
        exit()

    # Step 2: 准备时间戳列以进行时间序列操作

    # 提取每个间隔的起始时间点（分隔符 '-' 之前的部分）
    # 使用更健壮的逻辑来防止解析错误（如之前讨论的）
    df['Start_Time_Str'] = df['Time_Interval'].apply(lambda x: x.split('-')[0].strip())

    # 将提取的起始时间字符串转换为 datetime 对象，并指定正确的格式，使用 errors='coerce' 作为安全网
    df['Start_Time'] = pd.to_datetime(
        df['Start_Time_Str'],
        format=DATE_FORMAT,
        errors='coerce'
    )

    # 清理：移除时间戳无法解析的行
    df.dropna(subset=['Start_Time'], inplace=True)

    # 将 'Start_Time' 设置为索引
    df = df.set_index('Start_Time')

    # Step 3: 按 'Country_Area' 和 'Production_Type' 分组，并进行时间重采样 (Resampling)
    # **【核心改进】**：在 resample 之前，将 'Production_Type' 添加到分组键中。
    # 这样，重采样操作就会对每种发电类型的每小时数据独立进行平均。
    df_hourly = df.groupby(['Country_Area', 'Production_Type']).resample('H')['Total_Generation'].mean().reset_index()

    # Step 4: 清理和准备输出数据

    # 重命名时间列和聚合列
    df_hourly.rename(
        columns={
            'Start_Time': 'Hourly_Time_Start',
            'Total_Generation': 'Average_Hourly_Generation'
        },
        inplace=True
    )

    # 筛选最终列，现在包含 'Generation_Type'
    df_hourly = df_hourly[['Hourly_Time_Start', 'Country_Area', 'Production_Type', 'Average_Hourly_Generation']]

    # Step 5: 保存结果
    df_hourly.to_csv(output_file, index=False, encoding='utf-8')

    print("-" * 50)
    print(f"✅ 成功！数据降采样和分组已完成。")
    print(f"每小时平均发电量（按类型分组）数据已保存到文件: {output_file}")
    print(f"最终表格列名: Hourly_Time_Start, Country_Area, Production_Type, Average_Hourly_Generation")
    print("-" * 50)

except FileNotFoundError:
    print(f"错误：文件未找到。请确保文件名 '{input_file}' 正确且文件存在。")
except Exception as e:
    print(f"处理过程中发生未知错误: {e}")
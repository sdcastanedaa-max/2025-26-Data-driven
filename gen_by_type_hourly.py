import os
import pandas as pd

'''input_folder = "generation_by_type_2025"
output_path = "Wind_2025.csv"

hourly_dfs = {}

for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)

        gen_type = file.replace("_GENERATION.csv", "")

        # Clean and convert time
        df['Time_Interval'] = df['Time_Interval'].str.split(' - ').str[0]
        df['Time_Interval'] = pd.to_datetime(df['Time_Interval'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
        df.dropna(subset=['Time_Interval'], inplace=True)
        df.set_index('Time_Interval', inplace=True)

        # Resample to hourly
        hourly_df = df['Generation_Value'].resample('h').sum().to_frame(name=gen_type)

        hourly_dfs[gen_type] = hourly_df

# Combine all generation types
combined_hourly_df = pd.concat(hourly_dfs.values(), axis=1)
combined_hourly_df.to_csv(output_path)
'''
import os
import pandas as pd

input_folder = "generation_by_type_2025"
output_path = "Gen_data/Wind_2025.csv"

target_file = "Wind_Onshore_15min.csv"

# 检查目标文件是否存在
if target_file in os.listdir(input_folder):
    hourly_dfs = {}

    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path)

            gen_type = file.replace("_GENERATION.csv", "").replace(".csv", "")

            # 清洗和转换时间
            df['MTU (CET/CEST)'] = df['MTU (CET/CEST)'].str.split(' - ').str[0]
            df['MTU (CET/CEST)'] = pd.to_datetime(df['MTU (CET/CEST)'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
            df.dropna(subset=['MTU (CET/CEST)'], inplace=True)
            df.set_index('MTU (CET/CEST)', inplace=True)

            # 重新采样为每小时数据
            hourly_df = df['Generation(MW)'].resample('h').sum().to_frame(name=gen_type)

            hourly_dfs[gen_type] = hourly_df

    # 合并所有类型的发电数据
    combined_hourly_df = pd.concat(hourly_dfs.values(), axis=1)
    combined_hourly_df.to_csv(output_path)

    print(f"✅ 检测到 {target_file}，已生成 {output_path}")
else:
    print(f"⚠️ 未检测到 {target_file}，程序未执行。")

import os
import pandas as pd

input_folder = "generation_by_type_2025"
output_path = "generation_hourly_wind.csv"

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

import pandas as pd

# === 配置 ===
files = ["open-meteo-39.96N4.02W512m.csv"]
columns_to_keep = [
    "hourly__time",
    "hourly__apparent_temperature",
    "hourly__wind_speed_100m",
    "hourly__wind_direction_100m",
    "daily__weather_code"
]

# === 定义 WMO 天气代码映射 ===
wmo_codes = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

# === 读取并合并 ===
dfs = []
for f in files:
    df = pd.read_csv(f)
    # 保留指定列
    df = df[columns_to_keep]
    dfs.append(df)

# 合并所有年份
combined = pd.concat(dfs, ignore_index=True)

# 转换 weather code
combined["daily__weather_code"] = combined["daily__weather_code"].map(wmo_codes)

# === 保存结果 ===
combined.to_csv("Wind_2022-2024.csv", index=False)

print("✅ 数据合并完成！输出文件：Wind_2022-2024.csv")

wind_df = pd.read_csv("../Gen_data/Wind_2022-2024.csv")
wind_df['hourly__time'] = pd.to_datetime(wind_df['hourly__time'], format="%Y-%m-%dT%H:%M")
wind_df.to_csv("Wind_2022-2024.csv", index=False)

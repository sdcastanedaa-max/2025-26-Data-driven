import pandas as pd

# === 配置 ===
input_file = "Wind_2025.csv"#"Wind_2022_2024_raw.csv"
output_file = "Wind_2025_new.csv"

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

# === 读取原始 CSV ===
df = pd.read_csv(input_file)

# === 统一列名为当前脚本格式（你的旧模型格式） ===
df = df.rename(columns={
    "time": "hourly__time",
    "apparent_temperature": "hourly__apparent_temperature",
    "wind_speed_100m": "hourly__wind_speed_100m",
    "weather_code": "hourly__weather_code"
})

# === 选择需要的列（根据你实际数据修正） ===
columns_to_keep = [
    "hourly__time",
    "hourly__apparent_temperature",
    "hourly__wind_speed_100m",
    "hourly__weather_code",
    "cloud_cover",
    "precipitation",
]

df = df[columns_to_keep]

# === 翻译 weather code ===
df["hourly__weather_code"] = df["hourly__weather_code"].map(wmo_codes)

# === 统一时间格式 ===
df["hourly__time"] = pd.to_datetime(df["hourly__time"], format="%Y-%m-%dT%H:%M")

# === 保存 ===
df.to_csv(output_file, index=False)

print("✅ 数据处理完成！已输出：", output_file)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

# 1. Load datasets
df_load = pd.read_csv("Load Data/TotalLoadDayAhead_Hour_2022_2024_combined.csv", parse_dates=["datetime"])
df_gen = pd.read_csv("Gen_data/average_hourly_generation.csv", parse_dates=["Hourly_Time_Start"])
df_types = pd.read_csv("Gen_data/gen_hourly_MW_all_wide.csv", parse_dates=["datetime"])

# 2. Rename + clean
df_gen.rename(columns={"Hourly_Time_Start": "datetime"}, inplace=True)
df_types.rename(columns={"Time_Interval": "datetime"}, inplace=True)

# Drop useless string columns
for df in [df_load, df_gen, df_types]:
    df.drop(columns=["Country_Area", "Area"], errors="ignore", inplace=True)

# 3. Set index and resample
df_load.set_index("datetime", inplace=True)
df_gen.set_index("datetime", inplace=True)
df_types.set_index("datetime", inplace=True)

daily_load = df_load.resample('D').mean(numeric_only=True)/1000
daily_gen = df_gen.resample('D').mean(numeric_only=True)/1000
daily_types = df_types.resample('D').mean(numeric_only=True)/1000



#Plot daily Load & Generation

plt.figure(figsize=(14, 5))
plt.plot(daily_load.index, daily_load["Actual Total Load (MW)"], label="Actual Load")
plt.plot(daily_load.index, daily_load["Day-ahead Total Load Forecast (MW)"], label="Forecast Load", linestyle='--')
plt.plot(daily_gen.index, daily_gen["Average_Hourly_Generation"], label="Total Generation", alpha=0.7)
plt.title("Daily Load vs Forecast vs Total Generation")
plt.xlabel("Date")
plt.ylabel("Power (GW)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/daily_load_vs_gen.png", dpi=300, bbox_inches='tight')
plt.show()


#Generation by type (stacked area chart)

#daily_types_sorted = daily_types[daily_types.columns.sort_values()]  # optional sort
daily_types.plot.area(stacked=True, figsize=(14, 6), alpha=0.85)
plt.title("Daily Generation by Type (Stacked Area)")
plt.xlabel("Date")
plt.ylabel("Power (GW)")
plt.tight_layout()
plt.savefig("plots/gen_type.png", dpi=300, bbox_inches='tight')
plt.show()


#Monthly average
monthly_load = df_load.resample('ME').mean()/1000
monthly_gen = df_gen.resample('ME').mean()/1000

monthly = pd.DataFrame({
    "Actual Load": monthly_load["Actual Total Load (MW)"],
    "Forecast Load": monthly_load["Day-ahead Total Load Forecast (MW)"],
    "Total Generation": monthly_gen["Average_Hourly_Generation"]
})

monthly.plot(kind='bar', figsize=(14, 6))
plt.title("Monthly Average: Load and Generation")
plt.ylabel("Power (GW)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/monthly_gen_load.png", dpi=300, bbox_inches='tight')
plt.show()


#Peak Hour Analysis

df_load["hour"] = df_load.index.hour
df_gen["hour"] = df_gen.index.hour

load_by_hour = df_load.groupby("hour")[["Actual Total Load (MW)"]].mean()/1000
gen_by_hour = df_gen.groupby("hour")[["Average_Hourly_Generation"]].mean()/1000

plt.figure(figsize=(10, 5))
plt.plot(load_by_hour.index, load_by_hour["Actual Total Load (MW)"], label="Load by Hour")
plt.plot(gen_by_hour.index, gen_by_hour["Average_Hourly_Generation"], label="Generation by Hour", linestyle="--")
plt.title("Average Load and Generation by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Power (GW)")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/Peak_analysis.png", dpi=300, bbox_inches='tight')
plt.show()


#Correlation heatmap
#When one variable changes, how likely is the other to change with it, and in which direction

plt.figure(figsize=(12, 10))
corr = daily_types.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Generation Types")
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

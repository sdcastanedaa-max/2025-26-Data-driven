import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as mtick

# --- Muted color palette centered on #75896b ---
palette = {
    "primary": "#75896b",  # key tone
    "blue":    "#6b7d91",
    "red":     "#a06b6b",
    "purple":  "#8a6b91",
    "amber":   "#b09b6b"
}

# --- Global Matplotlib style update ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Georgia", "DejaVu Serif", "Garamond"],
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.edgecolor": "#3f4739",
    "axes.labelcolor": "#3f4739",
    "xtick.color": "#3f4739",
    "ytick.color": "#3f4739",
    "text.color": "#3f4739",
    "axes.titlepad": 10,
    "grid.color": "#c8d1bd",
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
sns.set_palette([palette["primary"], palette["blue"], palette["red"], palette["purple"], palette["amber"]])

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

# Remove low-impact generation types (less than 1% of total energy)
total_energy = daily_types.sum()
share = total_energy / total_energy.sum()
daily_types_clean = daily_types.loc[:, share >= 0.01]

# --- Daily Load vs Generation ---
plt.figure(figsize=(14, 5))
plt.plot(daily_load.index, daily_load["Actual Total Load (MW)"], label="Actual Load", color=palette["blue"])
plt.plot(daily_load.index, daily_load["Day-ahead Total Load Forecast (MW)"], label="Forecast Load", linestyle='--', color=palette["red"])
plt.plot(daily_gen.index, daily_gen["Average_Hourly_Generation"], label="Total Generation", alpha=0.8, color=palette["primary"])
plt.title("Daily Load vs Forecast vs Total Generation")
plt.xlabel("Date")
plt.ylabel("Power (GW)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/daily_load_vs_gen.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Generation by Type (Stacked Area) ---
daily_types_clean.plot.area(stacked=True, figsize=(14, 6), alpha=0.9, colormap="Accent")
plt.title("Daily Generation by Type (Stacked Area)")
plt.xlabel("Date")
plt.ylabel("Power (GW)")
plt.tight_layout()
plt.savefig("plots/gen_type.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Generation shares (overall + over time) ---
os.makedirs("plots", exist_ok=True)

# Keep only the columns that survived low-impact filter
types_hourly = df_types.resample("h").mean(numeric_only=True).fillna(0)
types_hourly_clean = types_hourly[daily_types_clean.columns]

# Overall share across the whole period
energy_MWh = types_hourly.sum()
share_overall = (energy_MWh / energy_MWh.sum()).sort_values(ascending=False)
energy_MWh_clean = types_hourly_clean.sum()
share_overall_clean = (energy_MWh_clean / energy_MWh_clean.sum()).sort_values(ascending=False)

# Bar chart of overall shares
plt.figure(figsize=(12, 5))
share_overall.plot(kind="bar", color=palette["primary"])
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.ylabel("Share of Total Generation")
plt.title("Generation Mix Share (entire period)")
plt.tight_layout()
plt.savefig("plots/generation_share_bar.png", dpi=300, bbox_inches="tight")
plt.show()

# Bar chart of overall shares (clean)
plt.figure(figsize=(12, 5))
share_overall_clean.plot(kind="bar", color=palette["primary"])
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.ylabel("Share of Total Generation")
plt.title("Clean Generation Mix Share (entire period)")
plt.tight_layout()
plt.savefig("plots/generation_share_bar_clean.png", dpi=300, bbox_inches="tight")
plt.show()

# Pie chart
plt.figure(figsize=(7, 7))
share_overall.plot(kind="pie", autopct="%1.1f%%", colors=[palette["primary"], palette["blue"], palette["red"], palette["purple"], palette["amber"]], ylabel="")
plt.title("Generation Mix Share (entire period)")
plt.tight_layout()
plt.savefig("plots/generation_share_pie.png", dpi=300, bbox_inches="tight")
plt.show()

# Pie chart (clean)
plt.figure(figsize=(7, 7))
share_overall_clean.plot(kind="pie", autopct="%1.1f%%", colors=[palette["primary"], palette["blue"], palette["red"], palette["purple"], palette["amber"]], ylabel="")
plt.title("Clean Generation Mix Share (entire period)")
plt.tight_layout()
plt.savefig("plots/generation_share_pie_clean.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Time-varying share (weekly) ---
weekly_mean = types_hourly_clean.resample("W").mean()
weekly_share_pct = weekly_mean.div(weekly_mean.sum(axis=1), axis=0) * 100
weekly_share_pct.plot.area(stacked=True, figsize=(14, 6), alpha=0.9, colormap="Accent")
plt.ylabel("Share of Total Generation (%)")
plt.xlabel("Date")
plt.title("Generation Mix Share Over Time (weekly)")
plt.tight_layout()
plt.savefig("plots/generation_share_over_time.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Monthly averages ---
monthly_load = df_load.resample('ME').mean()/1000
monthly_gen = df_gen.resample('ME').mean()/1000
monthly = pd.DataFrame({
    "Actual Load": monthly_load["Actual Total Load (MW)"],
    "Forecast Load": monthly_load["Day-ahead Total Load Forecast (MW)"],
    "Total Generation": monthly_gen["Average_Hourly_Generation"]
})
monthly.plot(kind='bar', figsize=(14, 6), color=[palette["blue"], palette["amber"], palette["primary"]])
plt.title("Monthly Average: Load and Generation")
plt.ylabel("Power (GW)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/monthly_gen_load.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Peak Hour Analysis ---
df_load["hour"] = df_load.index.hour
df_gen["hour"] = df_gen.index.hour
load_by_hour = df_load.groupby("hour")[["Actual Total Load (MW)"]].mean()/1000
gen_by_hour = df_gen.groupby("hour")[["Average_Hourly_Generation"]].mean()/1000

plt.figure(figsize=(10, 5))
plt.plot(load_by_hour.index, load_by_hour["Actual Total Load (MW)"], label="Load by Hour", color=palette["blue"])
plt.plot(gen_by_hour.index, gen_by_hour["Average_Hourly_Generation"], label="Generation by Hour", linestyle="--", color=palette["primary"])
plt.title("Average Load and Generation by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Power (GW)")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/Peak_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(12, 10))
corr = daily_types_clean.corr()
sns.heatmap(
    corr, annot=True,
    cmap=sns.diverging_palette(145, 15, as_cmap=True),  # gentle muted diverging palette
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap: Generation Types")
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
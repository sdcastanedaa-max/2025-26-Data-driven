import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# --- Muted color palette centered on #75896b ---
palette = {
    "primary": "#75896b",  # your key color
    "blue":    "#6b7d91",
    "red":     "#a06b6b",
    "purple":  "#8a6b91",
    "amber":   "#b09b6b"
}

# Update global Matplotlib style
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

# ---------- Config ----------
data_dir = Path("Load Data")

# Input filenames
f_2023_15 = data_dir / "TotalLoadDayAhead_15min_2023.csv"
f_2024_15 = data_dir / "TotalLoadDayAhead_15min_2024.csv"
f_2022_hr = data_dir / "TotalLoadDayAhead_Hour_2022.csv"

# Output filenames (saved in the same folder)
out_2023_hr = data_dir / "TotalLoadDayAhead_Hour_2023.csv"
out_2024_hr = data_dir / "TotalLoadDayAhead_Hour_2024.csv"
out_2022_aligned = data_dir / "TotalLoadDayAhead_Hour_2022_aligned.csv"

# Possible datetime column names from ENTSO-E
POSSIBLE_TIME_COLS = ["MTU (CET/CEST)", "MTU (UTC)", "datetime"]

def load_and_clean_time_col(df: pd.DataFrame) -> pd.DataFrame:
    """Find and clean the datetime column, parse as datetime, and set as index."""
    time_col = next((col for col in POSSIBLE_TIME_COLS if col in df.columns), None)
    if time_col is None:
        raise ValueError(f"Could not find a time column in {POSSIBLE_TIME_COLS}")

    df = df.rename(columns={time_col: "datetime"}).copy()
    df["datetime"] = (
        df["datetime"]
        .astype(str)
        .str.replace(r"\s*\(CET\)|\s*\(CEST\)|\s*\(UTC\)", "", regex=True)
        .str.split(" - ").str[0]  # take the start of the interval
        .str.strip()
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime")
    return df

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns (except datetime) to numeric, coercing errors to NaN."""
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def collapse_duplicates(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """
    Collapse duplicate timestamps (e.g., at DST fallback).
    how='mean' for MW-like series, 'sum' if you have energy quantities.
    """
    dups = df.index.duplicated(keep=False).sum()
    if dups:
        print(f"⚠️ Found {dups} duplicate timestamp rows; collapsing by {how}.")
        if how == "sum":
            df = df.groupby(level=0).sum()
        else:
            df = df.groupby(level=0).mean()
    return df

def process_15min_to_hour(in_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    df = load_and_clean_time_col(df)
    df = coerce_numeric(df)

    # Resample to hourly mean (use .sum() if your columns are energy per 15-min)
    df_hr = df.resample("h").mean().reset_index()
    df_hr.to_csv(out_path, index=False)
    print(f"✅ Hourly resampling done: {in_path.name} → {out_path.name}")
    return df_hr

def clean_hourly_file(in_path: Path) -> pd.DataFrame:
    """Normalize and clean an already-hourly file."""
    df = pd.read_csv(in_path)
    df = load_and_clean_time_col(df)
    df = coerce_numeric(df)

    # 1) Collapse any duplicate hourly timestamps first (DST fallback etc.)
    df = collapse_duplicates(df, how="mean")  # change to "sum" if appropriate

    # 2) Now it's safe to force a perfect hourly grid
    df = df.asfreq("h").reset_index()
    return df

def align_headers(df: pd.DataFrame, ref_cols: list) -> pd.DataFrame:
    """Match columns to reference order, add missing ones as NaN."""
    if ref_cols[0] != "datetime":
        ref_cols = ["datetime"] + [c for c in ref_cols if c != "datetime"]

    for col in ref_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[ref_cols]
    return df

def main():
    df_2023_hr = process_15min_to_hour(f_2023_15, out_2023_hr)
    df_2024_hr = process_15min_to_hour(f_2024_15, out_2024_hr)

    # Make a canonical schema from 2023 + 2024
    ref_cols = list(df_2023_hr.columns)
    for c in df_2024_hr.columns:
        if c not in ref_cols:
            ref_cols.append(c)

    df_2022_clean = clean_hourly_file(f_2022_hr)
    df_2022_aligned = align_headers(df_2022_clean, ref_cols)
    df_2022_aligned.to_csv(out_2022_aligned, index=False)
    print(f"✅ 2022 aligned: {f_2022_hr.name} → {out_2022_aligned.name}")

    print("\n=== Summary ===")
    for name, df in [("2023", df_2023_hr), ("2024", df_2024_hr), ("2022 (aligned)", df_2022_aligned)]:
        print(f"{name}: {len(df)} rows, columns: {list(df.columns)}")
        print(df.head(3), "\n")

    # 5) Combine all three aligned dataframes into one
    df_all = pd.concat([df_2022_aligned, df_2023_hr, df_2024_hr], ignore_index=True)
    df_all.to_csv(data_dir / "TotalLoadDayAhead_Hour_2022_2024_combined.csv", index=False)
    print("✅ Combined file saved: TotalLoadDayAhead_Hour_2022_2024_combined.csv")

if __name__ == "__main__":
    main()
# Path to combined file
data_dir = Path("Load Data")
plots_dir = Path("plots")
combined_file = data_dir / "TotalLoadDayAhead_Hour_2022_2024_combined.csv"


# Load and prepare
df_all = pd.read_csv(combined_file, parse_dates=["datetime"])
df_all = df_all.sort_values("datetime")

# Try to auto-detect column names
cols = [c.lower() for c in df_all.columns]

# Common ENTSO-E names: 'Total Load - Day Ahead [MW]', 'Total Load - Actual [MW]'
col_actual = next((c for c in df_all.columns if "actual" in c.lower()), None)
col_forecast = next((c for c in df_all.columns if "day" in c.lower() or "forecast" in c.lower()), None)

if not col_actual or not col_forecast:
    raise ValueError(f"Couldn’t find expected 'actual' and 'day-ahead' columns. Columns found: {df_all.columns.tolist()}")

# --- Plot full period ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_all["datetime"], df_all[col_actual],
        color=palette["primary"], label="Actual Load")
ax.plot(df_all["datetime"], df_all[col_forecast],
        color=palette["blue"], label="Day-Ahead Forecast", alpha=0.8, linewidth=1.6)

ax.set_title("Electric Load – Actual vs Day-Ahead Forecast (2022-2024)")
ax.set_xlabel("Date")
ax.set_ylabel("Load [MW]")
ax.grid(True)
ax.legend(frameon=False)

plt.tight_layout()
fig.savefig(plots_dir / "TotalLoad_Actual_vs_DayAhead_2022_2024.png", dpi=300)

# --- Optional: Zoom on a single period for clarity ---
start, end = "2023-01-01", "2023-12-01"
mask = (df_all["datetime"] >= start) & (df_all["datetime"] < end)
df_zoom = df_all.loc[mask]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_zoom["datetime"], df_zoom[col_actual],
        color=palette["primary"], label="Actual Load")
ax.plot(df_zoom["datetime"], df_zoom[col_forecast],
        color=palette["blue"], label="Day-Ahead Forecast", alpha=0.8)

ax.set_title(f"Zoomed View: {start} to {end}")
ax.set_xlabel("Date")
ax.set_ylabel("Load [MW]")
ax.legend(frameon=False)
ax.grid(True)

plt.tight_layout()
fig.savefig(plots_dir / f"TotalLoad_Zoom_{start}_to_{end}.png", dpi=300)
plt.show()

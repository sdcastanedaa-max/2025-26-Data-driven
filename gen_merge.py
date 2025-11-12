import pandas as pd
import glob
import os


def merge_energy_data(input_folder, output_file):

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        print("NO CSV FILES FOUND")
        return

    print(f"FIND {len(csv_files)} FILE(s), MERGING")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")

    # read and merge
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # no duplicate
    merged = merged.drop_duplicates(subset=["datetime", "Area", "Production Type"], keep="last")

    # sort
    merged = merged.sort_values(by="datetime")

    # save
    merged.to_csv(output_file, index=False)
    print(f"SAVED TO:{output_file}")
    print(f"TOTAL COLUMN:{len(merged)}")

# EXAMPLE
# merge_energy_data("data_folder", "merged_energy_data.csv")

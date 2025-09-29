import pandas as pd
import glob
import os

# Get all CSV files in the current directory
csv_files = glob.glob("*.csv")

for file in csv_files:
    df = pd.read_csv(file)
    # Select first two columns and the last column
    selected = df.iloc[:, [0, 1, -1]]
    # Rename columns
    selected.columns = ['id', 'question', 'translation']
    # Save back to CSV (overwrite)
    selected.to_csv(file, index=False)
import pandas as pd
import random
import os

# Paths
input_csv = r'D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\translated-csv\Sylheti_human.csv'
output_dir = r'D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples'
output_csv = os.path.join(output_dir, 'Test_Samples_Sylheti_Human.csv')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the CSV
df = pd.read_csv(input_csv)

# Select first 800 samples
df_800 = df.iloc[:800]

# Randomly select 500 samples
selected_df = df_800.sample(n=500, random_state=42)

# Save to CSV
selected_df.to_csv(output_csv, index=False)
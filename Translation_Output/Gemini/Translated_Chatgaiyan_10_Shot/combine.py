import pandas as pd
import os

def combine_train_csvs():
    """
    Combine train CSV files from train0.csv to train9.csv in order.
    """
    # List to store all dataframes
    dfs = []
    
    # Process files from train0 to train9
    for i in range(10):
        filename = f'train{i}.csv'
        
        # Check if file exists
        if os.path.exists(filename):
            print(f"Reading {filename}...")
            
            # Read the CSV file
            df = pd.read_csv(filename, header=None)
            dfs.append(df)
            
            print(f"  - Shape: {df.shape}")
        else:
            print(f"Warning: {filename} not found!")
    
    if dfs:
        # Combine all dataframes
        print("\nCombining all dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save the combined dataframe
        output_filename = 'combined_train.csv'
        combined_df.to_csv(output_filename, index=False, header=False)
        
        print(f"\nCombined dataset saved as '{output_filename}'")
        print(f"Final shape: {combined_df.shape}")
        print(f"Total rows: {len(combined_df)}")
        
        # Show first few rows
        print("\nFirst 5 rows of combined dataset:")
        print(combined_df.head())
        
        return combined_df
    else:
        print("No train files found to combine!")
        return None

if __name__ == "__main__":
    # Run the combination function
    combined_data = combine_train_csvs()

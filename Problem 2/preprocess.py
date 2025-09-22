import os
import pandas as pd
import numpy as np

def preprocess_trace_file(input_path, output_folder, col_index=0):
    print(f"Preprocessing {input_path}")
    # Read CSV: skip comments, handle whitespace, ignore bad lines
    df = pd.read_csv(input_path, header=None, comment='#', sep='\s+', on_bad_lines='skip')
    
    # Remove commas from numeric columns (especially counts)
    df[col_index] = df[col_index].astype(str).str.replace(',', '', regex=False)
    
    # Convert columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows that are completely NaN
    df.dropna(how='all', inplace=True)
    
    # Fill remaining NaNs
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Optional: normalize the column to match training scale (0-1)
    col_values = df[col_index].values.astype(float)
    if len(col_values) > 0:
        col_min = col_values.min()
        col_max = col_values.max()
        if col_max - col_min > 0:
            df[col_index] = (col_values - col_min) / (col_max - col_min)
    
    print(f"Cleaned data shape: {df.shape}")
    if df.empty:
        print("Warning: No data left after cleaning!")
    
    # Save cleaned CSV
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    clean_path = os.path.join(output_folder, os.path.basename(input_path))
    df.to_csv(clean_path, index=False, header=False)
    print(f"Saved cleaned data to {clean_path}")

def preprocess_all_traces(input_folder='ceg_data', output_folder='new_traces_clean', col_index=0):
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.csv'):
            preprocess_trace_file(os.path.join(input_folder, filename), output_folder, col_index)

if __name__ == "__main__":
    preprocess_all_traces()

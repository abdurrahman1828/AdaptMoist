import pandas as pd

data_name = 'enviva_b2'
# List of file paths for the CSV files to be combined
csv_files = [f'haralick_features_{data_name}.csv',
             f'fos_features_{data_name}.csv',
             f'fps_features_{data_name}.csv',
             f'glrlm_features_{data_name}.csv',
             f'lbp_features_{data_name}.csv']

# Read the first CSV file to initialize the final DataFrame
combined_df = pd.read_csv(csv_files[0])

# Loop through the remaining CSV files and concatenate them horizontally
for file in csv_files[1:]:
    df = pd.read_csv(file)
    combined_df = pd.concat([combined_df, df], axis=1)

# Save the final combined DataFrame to a new CSV file
combined_df.to_csv(f'combined_features_{data_name}.csv', index=False)

print("CSV files have been combined and saved")
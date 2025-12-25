import os
import pandas as pd
import glob

# Constants
SOURCE_DIR = "dataset_currency/csv_output"
OUTPUT_DIR = "dataset/global_macro"
YEARS = range(2005, 2016) # 2005 to 2015 inclusive
ASSETS = [
    'EUR_USD', 'GBP_USD', 'SPX500_USD', 'JP225_USD', 
    'XAU_USD', 'WTICO_USD', 'USB10Y_USD', 'CORN_USD'
]

def load_monthly_data(asset, year):
    """
    Loads and combines all monthly CSVs for a specific asset and year.
    Pattern: dataset_currency/csv_output/{ASSET}/{YEAR}/oanda-{ASSET}-{YEAR}-{MONTH}.csv
    """
    year_dir = os.path.join(SOURCE_DIR, asset, str(year))
    if not os.path.exists(year_dir):
        print(f"Warning: Directory not found {year_dir}")
        return None
        
    # Pattern matching for months 1-12
    # e.g. oanda-EUR_USD-2005-1.csv
    all_files = glob.glob(os.path.join(year_dir, f"oanda-{asset}-{year}-*.csv"))
    
    if not all_files:
        print(f"Warning: No files found for {asset} in {year}")
        return None
        
    dfs = []
    for file in all_files:
        try:
            # Check if header exists or implies structure. 
            # Based on previous exploration, it seems fairly standard but let's be safe.
            # Assuming standard columns from Oanda output: Date, Open, High, Low, Close, Volume (maybe)
            df = pd.read_csv(file)
            
            # Ensure Date parsing
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif 'time' in df.columns: 
                # Check if 'Date' already exists to prevent duplication
                if 'Date' in df.columns:
                     df.drop(columns=['Date'], inplace=True)
                df['Date'] = pd.to_datetime(df['time'])
                df.drop(columns=['time'], inplace=True) # usage 'time' as source, then drop
                
            # Drop any existing index cols if they got read in
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)

            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if not dfs:
        return None
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates
    if 'Date' in combined_df.columns:
        combined_df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
        combined_df.sort_values('Date', inplace=True)
        
    combined_df.reset_index(drop=True, inplace=True)
    
    return combined_df

def process_all_assets():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for asset in ASSETS:
        print(f"Processing {asset}...")
        asset_merged_dfs = []
        
        # Create Asset Folder in Output
        asset_out_dir = os.path.join(OUTPUT_DIR, asset)
        if not os.path.exists(asset_out_dir):
            os.makedirs(asset_out_dir)
            
        for year in YEARS:
            df = load_monthly_data(asset, year)
            if df is not None:
                # Save Yearly File
                out_path = os.path.join(asset_out_dir, f"{year}.csv")
                df.to_csv(out_path, index=False)
                # print(f"  Saved {out_path} ({len(df)} rows)")
                
                asset_merged_dfs.append(df)
            else:
                print(f"  Missing data for {asset} {year}")

        # Save Merged File
        if asset_merged_dfs:
            full_df = pd.concat(asset_merged_dfs, ignore_index=True)
            full_df.sort_values('Date', inplace=True)
            full_df.reset_index(drop=True, inplace=True)
            
            merged_path = os.path.join(OUTPUT_DIR, f"{asset}_merged.csv")
            full_df.to_csv(merged_path, index=False)
            print(f"FAILED: None" if len(full_df) == 0 else f"SUCCESS: Saved {merged_path} ({len(full_df)} rows)")
        else:
            print(f"ERROR: No data found for {asset} across all years!")

if __name__ == "__main__":
    process_all_assets()

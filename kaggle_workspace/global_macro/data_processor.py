import pandas as pd
import numpy as np
import pandas_ta as ta

def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    Expects columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
    """
    try:
        # Check if file has header or not. 
        # For merged files (dataset/ready), headers exist. For original files, maybe not.
        # Simple heuristic: Try reading with header=0. If columns don't match expected, try header=None.
        
        df = pd.read_csv(filepath)
        
        expected_cols = {'Open', 'High', 'Low', 'Close'}
        if not expected_cols.issubset(df.columns):
            # Fallback for old headerless files
            print(f"Header mismatch in {filepath}, trying headers=None...")
            df = pd.read_csv(filepath, header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Force numeric columns immediately to handle potential data corruption
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with bad numeric data
        df.dropna(subset=[c for c in cols if c in df.columns], inplace=True)

        # Try to parse Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Remove duplicate indices to prevent pandas-ta errors
            df = df[~df.index.duplicated(keep='first')]
            
        print(f"Loaded data head:\n{df.head()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def add_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame using Pandas-TA.
    """
    # Ensure data is float
    # Pandas-TA works directly on the DataFrame columns
    
    # Momentum Indicators
    df['RSI'] = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        df.rename(columns={
            'MACD_12_26_9': 'MACD', 
            'MACDs_12_26_9': 'MACD_SIGNAL',
            'MACDh_12_26_9': 'MACD_HIST'
        }, inplace=True)
    
    df['CCI'] = df.ta.cci(length=20)
    df['ADX'] = df.ta.adx(length=14)['ADX_14']

    # Overlap Studies (SMA/EMA)
    df['SMA_20'] = df.ta.sma(length=20)
    df['SMA_50'] = df.ta.sma(length=50)
    df['SMA_200'] = df.ta.sma(length=200) # Added for long-term trend context
    df['EMA_12'] = df.ta.ema(length=12)
    df['EMA_26'] = df.ta.ema(length=26)
    
    # Volatility
    df['ATR'] = df.ta.atr(length=14)
    
    bbands = df.ta.bbands(length=20, std=2)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
        # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        df.rename(columns={
            'BBL_20_2.0': 'BBL',
            'BBM_20_2.0': 'BBM',
            'BBU_20_2.0': 'BBU'
        }, inplace=True)
        
    # Volume
    df['OBV'] = df.ta.obv()

    # Drop NaNs created by indicators (e.g., first 50 rows)
    df.dropna(inplace=True)
    return df

def clean_data(df):
    """
    Removes infinite values and fills missing ones.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

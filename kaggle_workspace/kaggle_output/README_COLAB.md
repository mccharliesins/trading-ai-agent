# Trading AI Agent - Google Colab Setup

This folder contains everything you need to run the Trading AI Agent in Google Colab.

## Instructions

1.  **Upload to Drive:**
    *   Upload this entire `colab_workspace` folder to your Google Drive (e.g., at `/content/drive/MyDrive/colab_workspace`).
    *   **CRITICAL:** You must also upload the `dataset` folder from your desktop to `colab_workspace/dataset`. The scripts expect data at `dataset/ready/SPX500_USD`.

2.  **Open the Notebook:**
    *   Open `TradingBot.ipynb` in Google Colab.

3.  **Run Cells:**
    *   The notebook handles dependency installation.
    *   It mounts your Google Drive.
    *   It runs `train.py` to train the model.
    *   It runs `backtest.py` to verify results.

## File Structure Verification
Ensure your Drive folder looks like this:
```
/colab_workspace
  /dataset
     /ready
        /SPX500_USD
           SPX500_USD_2019.csv
           ...
  stock_env.py
  train.py
  backtest.py
  requirements.txt
  TradingBot.ipynb
```

# Trading AI Agent - Kaggle Deployment Guide

This guide explains how to deploy and run your Trading AI Agent on Kaggle Notebooks.

## Prerequisites

1.  A Kaggle Account.
2.  Your project files (local).

## Step 1: Upload Your Code as a Dataset

1.  Go to [Kaggle Datasets](https://www.kaggle.com/datasets) and click **New Dataset**.
2.  Drag and drop the contents of your `colab_workspace` folder (excluding the `dataset` folder if it's large, but INCLUDING `train.py`, `backtest.py`, `stock_env.py`, `data_processor.py`, `requirements.txt`).
3.  Name the dataset: `trading-bot-code`.
4.  Create the dataset.

## Step 2: Ensure Your Data is on Kaggle

1.  You mentioned your data is already at `/kaggle/input/readydataset`.
2.  Ensure this dataset contains the CSV files (e.g., `SPX500_USD_2015.csv`, etc.).

## Step 3: Create and Run the Notebook

1.  Go to [Kaggle Kernels/Code](https://www.kaggle.com/code) and click **New Notebook**.
2.  **Add Inputs**:
    *   Click **Add Input** (right sidebar).
    *   Search for your `trading-bot-code` dataset and add it.
    *   Search for your data dataset (the one mapped to `/kaggle/input/readydataset`) and add it.
3.  **Upload Notebook**:
    *   File -> Import Notebook.
    *   Upload `kaggle_workspace/TradingBot_Kaggle.ipynb` from this folder.
4.  **Enable Internet**:
    *   In the sidebar on the right, under "Notebook Options" (or "Settings"), verify that **Internet** is set to **On**.
    *   This is required to install dependencies (`pip install`).
5.  **Run All**:
    *   Click "Run All".

## Output

*   The trained model and backtest graphs will be saved to the `/kaggle/working` directory.
*   You can download them from the "Output" section of the notebook viewer on the right side.

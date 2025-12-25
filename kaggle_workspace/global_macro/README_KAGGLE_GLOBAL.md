# Global Macro Training on Kaggle

This guide explains how to deploy the 8-Asset Global Macro Training (100M Steps) to Kaggle.

## Step 1: Upload the Data
Kaggle needs the processed data. 
1.  Go to **Kaggle > Datasets > New Dataset**.
2.  Upload the **entire folder** `dataset/global_macro` from your local machine.
    *   It should contain folders like `EUR_USD`, `GBP_USD` and files like `EUR_USD_merged.csv`.
3.  Name the dataset: `global-macro-data-2005-2015`.
4.  Create it (Private is fine).

## Step 2: Create the Notebook
1.  Go to **Kaggle > Code > New Notebook**.
2.  **Add Data**: Click "+ Add Input" (top right) -> "Your Datasets" -> Select `global-macro-data-2005-2015`.
3.  **Accelerator**: Set to **GPU P100** or **T4 x2** (though this code uses CPU `SubprocVecEnv` efficiently, GPU can help the PPO optimization). **CPU-only** with many cores is also very good for 8-env parallelization.

## Step 3: Deployment (The Code)
You have two options:

### Option A: Upload Files (Recommended)
1.  Upload the files in this folder (`train_global_kaggle.py`, `stock_env.py`, `data_processor.py`) as a **Dataset** called `global-macro-codes`.
2.  Add that dataset to your notebook.
3.  Copy them to the working directory in the first cell:
    ```python
    !cp /kaggle/input/global-macro-codes/*.py ./
    ```

### Option B: Copy-Paste
Create 3 code cells and verify the filenames match these exactly:

**Cell 1: `data_processor.py`**
*   Paste content of `data_processor.py`.
*   Add `%%writefile data_processor.py` at the very top.

**Cell 2: `stock_env.py`**
*   Paste content of `stock_env.py`.
*   Add `%%writefile stock_env.py` at the very top.

**Cell 3: `train_global.py`**
*   Paste content of **`train_global_kaggle.py`** (from this folder).
*   **NOTE**: This file is already pre-configured with the correct Kaggle Input Path.
*   Add `%%writefile train_global.py` at the very top.

## Step 4: Run Training
In a new cell:

```python
# 1. Install Dependencies
!pip install stable-baselines3 shimmy>=0.2.1

# 2. Run
!python train_global.py
```

## Step 5: Download Model
Training will take hours (approx 10-20 hours for 100M steps).
When done:
1.  Check the `checkpoints/global_macro` folder in the Output.
2.  Download `ppo_global_macro_final.zip`.

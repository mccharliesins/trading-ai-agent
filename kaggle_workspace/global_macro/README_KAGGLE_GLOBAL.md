# Global Macro Trading on Kaggle (100M Steps)

This guide explains how to deploy the **8-Asset Global Macro Training** to Kaggle. This experiment trains a single PPO agent to trade uncorrelated assets simultaneously, learning universal market principles.

## Technical Specifications
*   **Algorithm**: PPO (Proximal Policy Optimization)
*   **Timesteps**: 100,000,000 (100 Million)
*   **Assets**: 8 Uncorrelated Instruments:
    *   Currencies: `EUR_USD`, `GBP_USD`
    *   Indices: `SPX500_USD`, `JP225_USD`
    *   Commodities: `XAU_USD` (Gold), `WTICO_USD` (Oil), `CORN_USD`
    *   Bonds: `USB10Y_USD`
*   **Training Period**: 2005 - 2015 (Includes 2008 Financial Crisis)
*   **Architecture**:
    *   Policy Net: `[128, 128]`
    *   Value Net: `[128, 128]`
    *   Normalization: `VecNormalize` (Obs & Reward)
*   **Risk Profile**:
    *   Leverage: **20x** (Futures style)
    *   Stop Loss: **1%**
    *   Take Profit: **3%** (1:3 Risk/Reward Ratio)
    *   Execution: Start Randomly in history (`random_start=True`)

---

## Step 1: Upload the Data
Kaggle needs the processed price data.
1.  Go to **Kaggle > Datasets > New Dataset**.
2.  Upload the **entire folder** `dataset/global_macro` from your local machine.
    *   It must contain `EUR_USD_merged.csv`, `SPX500_USD_merged.csv`, etc.
    *   **Crucial**: The columns should include `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
3.  Name the dataset: `global-macro-data`.
4.  Create it.

## Step 2: Upload the Code
Kaggle needs the scripts (`train_global_kaggle.py`, `stock_env.py`, `data_processor.py`).
1.  Go to **Kaggle > Datasets > New Dataset**.
2.  Upload the files in **this folder** (`kaggle_workspace/global_macro/`).
3.  Name the dataset: `global-macro-codes`.
    *   *(Note the plural 'codes')*
4.  Create it.

## Step 3: Create the Notebook
1.  Go to **Kaggle > Code > New Notebook**.
2.  **Add Data**: Click "+ Add Input" (top right).
    *   Add `global-macro-data`.
    *   Add `global-macro-codes`.
3.  **Accelerator**:
    *   **GPU P100** (Recommended for PPO optimization).
    *   **T4 x2** (Also good).
    *   *CPU-only* (viable for 8-env parallelization, but slower on gradient updates).

## Step 4: Run Training
Copy and paste this into the first cell of your notebook. This script automatically fixes column headers and sets up the environment.

```python
# 1. Install Dependencies
!pip install stable-baselines3 shimmy>=0.2.1 protobuf==3.20.3 pandas_ta

# 2. Copy Code to Working Directory
!cp /kaggle/input/global-macro-codes/*.py ./

# 3. Overwrite with Fixes (Just to be sure)
%%writefile train_global_kaggle.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import os

from stock_env import StockTradingEnv
from data_processor import add_technical_indicators, clean_data, load_data

def make_env(rank, seed=0):
    def _init():
        assets = ['EUR_USD', 'GBP_USD', 'SPX500_USD', 'JP225_USD', 'XAU_USD', 'WTICO_USD', 'USB10Y_USD', 'CORN_USD']
        asset_name = assets[rank % len(assets)]
        file_path = f"/kaggle/input/global-macro-data/{asset_name}_merged.csv"
        
        print(f"[Rank {rank}] Loading {asset_name} from {file_path}...")
        
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"{file_path}")
             
        df = pd.read_csv(file_path)
        
        # FIX 1: Renaming columns to Title Case
        df.columns = [c.capitalize() for c in df.columns]

        # FIX 2: Set Date as Index (Critical)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        df = add_technical_indicators(df)
        df = clean_data(df)
        
        env = StockTradingEnv(
             df=df, 
             commission_pct=0.0001, 
             leverage=20.0, 
             random_start=True,
             stop_loss_pct=0.01, 
             take_profit_pct=0.03 
        )
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    os.makedirs("checkpoints/global_macro", exist_ok=True)
    os.makedirs("logs/global_macro", exist_ok=True)
    
    num_cpu = 8 
    print(f"Starting Global Macro Training on {num_cpu} cores...")
    
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_callback = EvalCallback(
        env, 
        best_model_save_path='./checkpoints/global_macro/best_model',
        log_path='./logs/global_macro',
        eval_freq=100_000,
        deterministic=True,
        render=False
    )

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        ent_coef=0.01, 
        learning_rate=0.0003,
        batch_size=512,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs_global"
    )
    
    TOTAL_TIMESTEPS = 100_000_000
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
        model.save("checkpoints/global_macro/ppo_global_macro_final")
        env.save("checkpoints/global_macro/vec_normalize.pkl")
        print("Training Complete.")
    except Exception as e:
        print(f"Training interrupted: {e}")
        model.save("checkpoints/global_macro/ppo_global_macro_interrupted")
        env.save("checkpoints/global_macro/vec_normalize_interrupted.pkl")

# 4. Run Training
!python train_global_kaggle.py
```

## Step 5: Download Model
Training will take approx 10-20 hours.
When done:
1.  Check the `checkpoints/global_macro` folder.
2.  Download `ppo_global_macro_final.zip`.


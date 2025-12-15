import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
import matplotlib.pyplot as plt
import numpy as np

from data_processor import load_data, add_technical_indicators, clean_data
from stock_env import StockTradingEnv

# CONFIG
DATA_DIR = 'dataset/ready/SPX500_USD' # Update to point to the merged directory
TOTAL_TIMESTEPS = 5_000_000 
TRAIN_SPLIT = 0.9 
N_ENVS = 4
CONTRACT_SIZE = 1 # Default for CFD/Futures (1 Point = 1 Unit Currency). Adjust as needed (e.g. 50 for ES, 20 for NQ)
LEVERAGE = 30.0 # Standard Professional Leverage (1:30)

# Helper for Periodic Evaluation
class PeriodicTestCallback(BaseCallback):
    def __init__(self, check_freq: int, log_file: str, verbose=1):
        super(PeriodicTestCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_file = log_file
        self.years = [2016, 2017, 2018, 2019, 2020] # TEST Years
        self.data_dir = DATA_DIR
        self.graph_dir = 'training_graphs'
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)
        
        # Initialize Log File
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("| Step | Year | Total Trades | Profit % | Win Rate % | Graph |\n")
                f.write("|---|---|---|---|---|---|\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"\n--- Running Periodic Evaluation at Step {self.num_timesteps} ---")
            
            # Save checkpoint
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            checkpoint_name = f"checkpoints/model_{self.num_timesteps}_steps"
            self.model.save(checkpoint_name)
            
            # Evaluate on each year
            for year in self.years:
                filename = f"SPX500_USD_{year}.csv"
                filepath = os.path.join(self.data_dir, filename)
                
                if not os.path.exists(filepath):
                    continue
                
                try:
                    df = load_data(filepath)
                    if df is None: continue
                    df = add_technical_indicators(df)
                    df = clean_data(df)
                    df = df[~df.index.duplicated(keep='first')]

                    env = DummyVecEnv([lambda: StockTradingEnv(df, contract_size=CONTRACT_SIZE)])
                    obs = env.reset()
                    env_instance = env.envs[0]
                    
                    done = False
                    trades = 0
                    prev_shares = 0
                    
                    portfolio_history = []
                    
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, rewards, dones, _ = env.step(action)
                        done = dones[0]
                        
                        curr_shares = env_instance.shares_held
                        if curr_shares != prev_shares:
                            trades += 1
                        
                        prev_shares = curr_shares
                        portfolio_history.append(env_instance.net_worth)
                        
                    final_bal = env_instance.net_worth
                    initial_bal = env_instance.initial_balance
                    profit_pct = ((final_bal - initial_bal) / initial_bal) * 100
                    
                    # Generate Graph
                    plt.figure(figsize=(10,6))
                    plt.plot(portfolio_history)
                    plt.title(f"Eval {year} @ Step {self.num_timesteps} - Profit: {profit_pct:.2f}%")
                    plt.ylabel("Portfolio Value")
                    plt.xlabel("Step")
                    graph_path = f"{self.graph_dir}/eval_{self.num_timesteps}_{year}.png"
                    plt.savefig(graph_path)
                    plt.close()
                    
                    # Log result
                    with open(self.log_file, "a") as f:
                        f.write(f"| {self.num_timesteps} | {year} | {trades} | {profit_pct:.2f}% | N/A | [Graph]({graph_path}) |\n")
                        
                    print(f"Eval {year}: {profit_pct:.2f}% ({trades} trades)")
                    
                except Exception as e:
                    print(f"Error evaluating {year}: {e}")
            
        return True

def main():
    # 1. Load Training Datasets
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Directory {DATA_DIR} not found. Run merge_data.py first.")
        return

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    # Filter for Training Years (2005-2015)
    train_files = []
    for f in all_files:
        # Expected format: SPX500_USD_{YEAR}.csv
        try:
            year = int(f.split('_')[-1].split('.')[0])
            if 2005 <= year <= 2015:
                train_files.append(os.path.join(DATA_DIR, f))
        except ValueError:
            pass # Skip if format doesn't match
            
    if not train_files:
        print("No training files found for 2005-2015.")
        return

    print(f"Found {len(train_files)} training datasets (2005-2015).")
    
    # Pre-load dataframes
    all_dfs = []
    for f in train_files:
        print(f"Processing {f}...")
        df = load_data(f)
        if df is not None and not df.empty:
            try:
                df = add_technical_indicators(df)
                df = clean_data(df)
                if len(df) > 100: 
                    print(f"   -> Processed Shape: {df.shape} (Columns: {len(df.columns)})")
                    all_dfs.append(df)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")

                
    if not all_dfs:
        print("No valid dataframes loaded.")
        return

    # 2. Distribute Dataframes to Environments
    train_dfs = []
    val_dfs = []
    
    for df in all_dfs:
        split_idx = int(len(df) * TRAIN_SPLIT)
        train_dfs.append(df.iloc[:split_idx])
        val_dfs.append(df.iloc[split_idx:])
        
    def make_env(rank, is_train=True):
        def _init():
            # Assign a specific dataset to this environment based on its rank (ID)
            dataset_idx = rank % len(train_dfs)
            data = train_dfs[dataset_idx] if is_train else val_dfs[dataset_idx]
            return StockTradingEnv(data, contract_size=CONTRACT_SIZE, leverage=LEVERAGE)
        return _init

    # 3. Create Environments
    # SubprocVecEnv allows parallel training
    env_train = SubprocVecEnv([make_env(i, is_train=True) for i in range(N_ENVS)])
    # Normalize Observations and Rewards
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Validation env
    env_val = DummyVecEnv([make_env(0, is_train=False)])
    env_val = VecNormalize(env_val, norm_obs=True, norm_reward=False, clip_obs=10., training=False)

    # 4. Setup Callbacks
    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=15, min_evals=5, verbose=1)
    
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path='./best_model/',
        log_path='./logs/',
        eval_freq=50000,
        callback_after_eval=stop_callback,
        deterministic=True,
        render=False
    )

    # 5. Initialize Model
    # Optimize: Larger Net (128x128), Lower Learning Rate (3e-4 is default, maybe try 2e-4?), Higher Batch Size (256 default -> 512?)
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MlpPolicy", env_train, verbose=1, device="cpu", ent_coef=0.01, policy_kwargs=policy_kwargs, batch_size=512) 

    # 6. Periodic Callback
    # Instantiate it
    periodic_callback = PeriodicTestCallback(check_freq=20000, log_file="model_performance_log.md")

    # 7. Train
    print("Starting training...")
    try:
        # Combine callbacks
        # User requested to disable periodic evaluation during training
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback])
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    model.save("ppo_trading_agent_final")
    env_train.save("vec_normalize.pkl") # Save normalization stats
    print("Training finished. Model and Normalization stats saved.")

if __name__ == "__main__":
    main()

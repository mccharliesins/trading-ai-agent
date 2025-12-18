import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stock_env import StockTradingEnv
from data_processor import load_data, add_technical_indicators, clean_data
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

def debug_run():
    # Load 2019 Data
    # Load 2019 Data
    filepath = 'dataset/ready/SPX500_USD/SPX500_USD_2019.csv'
    df = load_data(filepath)
    df = add_technical_indicators(df)
    df = clean_data(df)
    
    # Slice to just 1 week (approx 7200 minutes) to see clear details
    df_slice = df.iloc[:7200] 
    
    print(f"Debugging on first {len(df_slice)} steps of 2019...")
    
    model_path = "best_model/best_model.zip"
    if not os.path.exists(model_path):
         model_path = "ppo_trading_agent_final.zip"
         
    model = PPO.load(model_path)
    env = DummyVecEnv([lambda: StockTradingEnv(df_slice, leverage=100.0)])
    
    # Load Normalization
    if os.path.exists("vec_normalize.pkl"):
        env = VecNormalize.load("vec_normalize.pkl", env)
        env.training = False
        env.norm_reward = False

    obs = env.reset()
    env_instance = env.envs[0]
    
    history = []
    actions = []
    
    done = False
    step = 0
    trades = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Manually track state changes
        prev_shares = env_instance.shares_held
        
        obs, reward, done, _ = env.step(action)
        
        curr_shares = env_instance.shares_held
        
        if curr_shares != prev_shares:
            type = "BUY" if curr_shares > prev_shares else "SELL"
            price = env_instance.df_values[env_instance.current_step][env_instance.close_idx]
            print(f"Step {step}: {type} at {price:.5f} | Balance: {env_instance.balance:.2f} | Action: {action}")
            trades += 1
            
        history.append(env_instance.net_worth)
        step += 1
        
    print(f"Total Trades in 1 week: {trades}")
    print(f"Final Net Worth: {history[-1]}")
    
    # Quick Plot
    plt.figure(figsize=(12,6))
    plt.plot(history, label='Net Worth')
    plt.title(f"Debug Run - 1 Week - {trades} Trades")
    plt.legend()
    plt.savefig("debug_graph.png")
    print("Saved debug_graph.png")

if __name__ == "__main__":
    debug_run()

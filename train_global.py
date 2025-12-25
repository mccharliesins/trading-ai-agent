import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import os

from stock_env import StockTradingEnv
from data_processor import add_technical_indicators, clean_data, load_data

def make_env(rank, seed=0):
    """
    Creates an environment for a specific asset based on the CPU rank.
    """
    def _init():
        # The "Global Macro" 8-Pack list
        assets = [
            'EUR_USD',      # 0
            'GBP_USD',      # 1
            'SPX500_USD',   # 2
            'JP225_USD',    # 3
            'XAU_USD',      # 4
            'WTICO_USD',    # 5
            'USB10Y_USD',   # 6
            'CORN_USD'      # 7
        ]
        
        # Select asset by rank
        asset_name = assets[rank % len(assets)]
        
        # Load merged data (created by prepare_global_data.py)
        file_path = f"dataset/global_macro/{asset_name}_merged.csv"
        
        print(f"[Rank {rank}] Loading {asset_name} from {file_path}...")
        
        if not os.path.exists(file_path):
             print(f"ERROR: File not found {file_path}. Is data prep done?")
             raise FileNotFoundError(f"{file_path}")
             
        df = pd.read_csv(file_path)
        
        # Preprocessing on the fly (or better, pre-process in step 1, but for now we do it here)
        # Note: If merged file is raw, we need to add indicators.
        # Assuming prepare_global_data just combined raw files.
        df = add_technical_indicators(df)
        df = clean_data(df)
        
        # Create environment with Random Start
        env = StockTradingEnv(
             df=df, 
             commission_pct=0.0001, 
             leverage=20.0, # Futures Leverage 1:20 (User Requested)
             random_start=True,
             stop_loss_pct=0.01, # 1% Risk
             take_profit_pct=0.03 # 3% Reward (1:3 Ratio)
        )
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    # Ensure output dirs
    os.makedirs("checkpoints/global_macro", exist_ok=True)
    os.makedirs("logs/global_macro", exist_ok=True)
    
    # Number of parallel environments = 8 assets
    num_cpu = 8 
    
    print(f"Starting Global Macro Training on {num_cpu} cores...")
    
    # Create the parallel environments
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # Normalize Observations and Rewards (CRITICAL adaptation from past model)
    # This helps the model handle different price scales (e.g. Gold 2000 vs EUR 1.05)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Setup Eval Callback to save best model during the long run
    # evaluating on training env for simplicity/speed in this massive run, 
    # ensuring we save progress even if we don't have separate hold-out set for all global assets handy yet
    eval_callback = EvalCallback(
        env, 
        best_model_save_path='./checkpoints/global_macro/best_model',
        log_path='./logs/global_macro',
        eval_freq=100_000,
        deterministic=True,
        render=False
    )

    # Define the model (PPO)
    # Refined Settings from past success:
    # - net_arch=[128, 128]: Deeper network for complex patterns
    # - batch_size=512: Better gradient estimation
    # - ent_coef=0.01: High exploration for multi-asset
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
    
    # Train
    TOTAL_TIMESTEPS = 100_000_000
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    print("Strategy: Global Macro (8 Assets) | Random Start | Normalized | Recurrent-Ready")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
        model.save("checkpoints/global_macro/ppo_global_macro_final")
        env.save("checkpoints/global_macro/vec_normalize.pkl") # Save normalization stats!
        print("Training Complete. Model and Normalization stats saved.")
    except Exception as e:
        print(f"Training interrupted or failed: {e}")
        model.save("checkpoints/global_macro/ppo_global_macro_interrupted")
        env.save("checkpoints/global_macro/vec_normalize_interrupted.pkl")

import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

from data_processor import load_data, add_technical_indicators, clean_data
from stock_env import StockTradingEnv

def backtest():
    DATA_DIR = 'dataset/ready/SPX500_USD' # Updated dataset
    # Logic to select model
    if os.path.exists("best_model/best_model.zip"):
        model_path = "best_model/best_model.zip"
    elif os.path.exists("ppo_trading_agent_final.zip"):
        model_path = "ppo_trading_agent_final.zip"
    else:
        print("No model found (checked best_model/best_model.zip and ppo_trading_agent_final.zip).")
        return

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print(f"DEBUG: Loaded Model Obs Space: {model.observation_space.shape}")
    
    # Years to test (2016-2020)
    years = [2016, 2017, 2018, 2019, 2020]
    
    CONTRACT_SIZE = 1 # Keep consistent with training
    LEVERAGE = 30.0
    
    for year in years:
        print(f"DEBUG: Processing {year}...", flush=True)
        filename = f"SPX500_USD_{year}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"Skipping {year}: File not found.")
            continue
            
        print(f"Backtesting on {year}...")
        df = load_data(filepath)
        df = add_technical_indicators(df)
        df = clean_data(df)
        
        env = DummyVecEnv([lambda: StockTradingEnv(df, contract_size=CONTRACT_SIZE, leverage=LEVERAGE)])
        
        # Load Normalization Stats
        if os.path.exists("vec_normalize.pkl"):
            print("Loading normalization stats...")
            env = VecNormalize.load("vec_normalize.pkl", env)
            env.training = False # Don't update stats during backtest
            env.norm_reward = False # See real rewards
        else:
            print("WARNING: vec_normalize.pkl not found. Running with raw observations (Results may be poor).")

        obs = env.reset()
        env_instance = env.envs[0]
        
        history_net_worth = []
        buy_signals = [] # (index, price)
        sell_signals = [] # (index, price)
        
        # Metric Trackers
        wins = 0
        losses = 0
        total_gain = 0
        total_loss = 0
        n_trades = 0
        
        done = False
        step = 0
        
        # Track previous shares to detect trades
        prev_shares = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # Capture Realized PnL from Info
            # VecEnv 'infos' is a list of dicts. We have 1 env.
            info = infos[0] 
            if 'pnl' in info and info['pnl'] != 0:
                pnl = info['pnl']
                n_trades += 1
                if pnl > 0:
                    wins += 1
                    total_gain += pnl
                else:
                    losses += 1
                    total_loss += abs(pnl)
            
            history_net_worth.append(env_instance.net_worth)
            
            # Record Buy/Sell for Plotting
            curr_shares = env_instance.shares_held
            current_price = env_instance.df_values[env_instance.current_step][env_instance.close_idx]
            
            # Simple trade detection
            if curr_shares > prev_shares: # Bought
                buy_signals.append((step, current_price))
            elif curr_shares < prev_shares: # Sold
                sell_signals.append((step, current_price))
                
            prev_shares = curr_shares
            step += 1
            done = dones[0]

        # Detailed Plotting for the Year
        print(f"Creating graph for {year}...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Calculate Metrics
        # 1. Total Return
        initial_balance = env_instance.initial_balance
        # Fix: Remove the last point if it dropped to initial balance due to reset
        if len(history_net_worth) > 1 and history_net_worth[-1] == 10000 and history_net_worth[-2] != 10000:
             history_net_worth.pop()
             
        final_balance = history_net_worth[-1]
        total_profit = final_balance - initial_balance
        return_pct = (total_profit / initial_balance) * 100
        
        # Metrics Math
        win_rate = (wins / n_trades * 100) if n_trades > 0 else 0.0
        avg_win = (total_gain / wins) if wins > 0 else 0.0
        avg_loss = (total_loss / losses) if losses > 0 else 0.0
        
        rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        profit_factor = (total_gain / total_loss) if total_loss > 0 else 0.0
        
        # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
        win_pct_decimal = wins / n_trades if n_trades > 0 else 0.0
        loss_pct_decimal = losses / n_trades if n_trades > 0 else 0.0
        expectancy = (win_pct_decimal * avg_win) - (loss_pct_decimal * avg_loss)

        # 3. Max Drawdown
        net_worth_series = pd.Series(history_net_worth)
        running_max = net_worth_series.cummax()
        drawdowns = (running_max - net_worth_series) / running_max
        max_drawdown = drawdowns.max()

        print("-" * 33)
        print(f"--- {year} Professional Metrics ---")
        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Balance:   ${final_balance:.2f}")
        print(f"Total Return:    ${total_profit:.2f} ({return_pct:.2f}%)")
        print(f"Total Trades:    {n_trades}")
        print(f"Win Rate:        {win_rate:.2f}%")
        print(f"Avg Win:         ${avg_win:.2f}")
        print(f"Avg Loss:        ${avg_loss:.2f}")
        print(f"Risk/Reward:     1:{rr_ratio:.2f}")
        print(f"Profit Factor:   {profit_factor:.2f}")
        print(f"Expectancy:      ${expectancy:.2f}")
        print(f"Max Drawdown:    -{max_drawdown*100:.2f}%")
        
        if expectancy > 5 and profit_factor > 1.2: # Loose pass for now
             print(">>> STATUS: PASS (Positive Expectancy)")
        else:
             print(">>> STATUS: FAIL (Negative Expectancy)")

        print("-" * 33)
        
        # 3. Max Drawdown
        net_worth_series = pd.Series(history_net_worth)
        rolling_max = net_worth_series.cummax()
        drawdown = (net_worth_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Print Stats to Console
        print(f"--- {year} Performance Report ---")
        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Balance:   ${final_balance:.2f}")
        print(f"Total Return:    ${total_profit:.2f} ({return_pct:.2f}%)")
        print(f"Total Trades:    {n_trades}")
        print(f"Win Rate:        {win_rate:.2f}%")
        print(f"Risk/Reward:     {rr_ratio:.2f}")
        print(f"Profit Factor:   {profit_factor:.2f}")
        print(f"Max Drawdown:    {max_drawdown:.2f}%")
        print("---------------------------------")
        
        # Add Text to Plot
        stats_text = (
            f"Return: {return_pct:.2f}%\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Drawdown: {max_drawdown:.2f}%"
        )
        ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
        
        # Top Plot: Price & Trades
        close_prices = df['Close'].values[:len(history_net_worth)]
        
        ax1.plot(close_prices, label='SPX500 Price', color='black', alpha=0.6) # Updated Label
        
        # Plot Buy Markers
        if buy_signals:
            buy_x, buy_y = zip(*buy_signals)
            ax1.scatter(buy_x, buy_y, marker='^', color='green', s=100, label='Buy Signal', zorder=5)
            
        # Plot Sell Markers
        if sell_signals:
            sell_x, sell_y = zip(*sell_signals)
            ax1.scatter(sell_x, sell_y, marker='v', color='red', s=100, label='Sell Signal', zorder=5)
            
        ax1.set_title(f"SPX500 {year} - Price & Trades")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True)
        
        # Bottom Plot: Portfolio Value
        ax2.plot(history_net_worth, label='Portfolio Value', color='blue')
        ax2.set_title(f"Portfolio Performance (Initial: ${env_instance.initial_balance})")
        ax2.set_ylabel("Value ($)")
        ax2.set_xlabel("Time (Minutes)")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        output_file = f"backtest_result_{year}_FIXED.png"
        plt.savefig(output_file)
        print(f"Saved {output_file}")
        plt.close()

if __name__ == "__main__":
    backtest()

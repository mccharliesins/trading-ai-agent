import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import numpy as np

from data_processor import load_data, add_technical_indicators, clean_data
from stock_env import StockTradingEnv

def backtest():
    DATA_DIR = os.getenv('TRADING_DATA_DIR', 'dataset/ready/SPX500_USD') # Updated dataset
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
        
        # New Metrics
        active_trade_regime = "Unknown" # "Trend" or "Range"
        regime_stats = {
            "Trend": {"wins": 0, "losses": 0, "total_gain": 0, "total_loss": 0, "n_trades": 0},
            "Range": {"wins": 0, "losses": 0, "total_gain": 0, "total_loss": 0, "n_trades": 0}
        }
        
        done = False
        step = 0
        
        # Track previous shares to detect trades
        prev_shares = 0
        
        while not done:
            # Detect Entry (Flat -> Position)
            # We must check env state BEFORE prediction/step (which we can't do easily with VecEnv wrappers hiding internal step)
            # But we can check `prev_shares` vs `env_instance.shares_held` from LAST loop.
            # Wait, `prev_shares` is updated at the END of this loop with `curr_shares`.
            # So `prev_shares` holds the state at the beginning of this step.
            
            # Prediction
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # --- Capture Result (Exit) ---
            info = infos[0] 
            if 'pnl' in info and info['pnl'] != 0:
                pnl = info['pnl']
                n_trades += 1
                
                # General Stats
                if pnl > 0:
                    wins += 1
                    total_gain += pnl
                else:
                    losses += 1
                    total_loss += abs(pnl)
                
                # Regime Stats (Use the regime recorded at Entry)
                # If active_trade_regime is Unknown (e.g. started in position?), default to current
                current_adx = df['ADX'].iloc[env_instance.current_step if env_instance.current_step < len(df) else -1]
                regime_key = active_trade_regime if active_trade_regime != "Unknown" else ("Trend" if current_adx > 25 else "Range")

                regime_stats[regime_key]["n_trades"] += 1
                if pnl > 0:
                    regime_stats[regime_key]["wins"] += 1
                    regime_stats[regime_key]["total_gain"] += pnl
                else:
                    regime_stats[regime_key]["losses"] += 1
                    regime_stats[regime_key]["total_loss"] += abs(pnl)
            
            history_net_worth.append(env_instance.net_worth)
            
            # --- Detect Entry for Next Trade ---
            curr_shares = env_instance.shares_held
            current_price = env_instance.df_values[env_instance.current_step][env_instance.close_idx]
            
            # Check for Entry (0 -> +/- N) or Reversal/Add
            # Simple Entry Detection: prev=0, curr!=0
            if prev_shares == 0 and curr_shares != 0:
                # We just entered!
                # Determine Regime
                # env_instance.current_step is the 'next' step usually after step(), so checking ADX at current - 1 might be 'decision' time
                # but ADX changes slowly, so current is fine.
                idx = env_instance.current_step if env_instance.current_step < len(df) else len(df)-1
                current_adx = df['ADX'].iloc[idx]
                active_trade_regime = "Trend" if current_adx > 25 else "Range"
                
                buy_signals.append((step, current_price)) # Assuming Buy for now for plot markers, strictly logic below
                
            # Refine Plot Markers
            if curr_shares > prev_shares: # Bought (Entry Long or Cover Short)
                 if prev_shares == 0: # New Long
                     pass # Handled above for signals
                     if not (prev_shares == 0 and curr_shares != 0): # Avoid duplicate append if logic assumes above
                         buy_signals.append((step, current_price))
                 else:
                     buy_signals.append((step, current_price)) # Adding or Cover
                     
            elif curr_shares < prev_shares: # Sold (Entry Short or Sell Long)
                 # Actually `buy_signals` list is used for GREEN markers, `sell_signals` for RED.
                 # Convention: Green = Buy Action, Red = Sell Action.
                 sell_signals.append((step, current_price))

            prev_shares = curr_shares
            step += 1
            done = dones[0]

        # --- Post-Loop Calculation ---
        
        # 1. Net Worth Series Analysis
        net_worth_series = pd.Series(history_net_worth)
        
        # Max Drawdown & Timed Drawdown
        rolling_max = net_worth_series.cummax()
        drawdowns = (net_worth_series - rolling_max) / rolling_max
        max_drawdown_pct = drawdowns.min() * 100
        
        # Drawdown Duration
        # Identify periods where (net_worth < rolling_max)
        in_drawdown = net_worth_series < rolling_max
        
        # Calculate lengths of consecutive True values
        # Group cumulative sum of False values (which act as resetters)
        g = (~in_drawdown).cumsum()
        # Filter only True values, group by g, count
        drawdown_periods = in_drawdown.groupby(g).sum()
        max_duration_steps = drawdown_periods.max() if not drawdown_periods.empty else 0
        
        # Convert steps to Time
        # Assuming M1 data (1 minute per step)
        max_duration_minutes = max_duration_steps 
        max_duration_days = max_duration_minutes / (60 * 24) # Approximate
        max_duration_hours = max_duration_minutes / 60
        
        # 2. Trade Frequency
        # Total time covered by DF in days
        start_date = df.index[0]
        end_date = df.index[-1]
        total_days = (end_date - start_date).days
        if total_days == 0: total_days = 1
        
        trades_per_day = n_trades / total_days
        avg_time_between_trades_hours = (24 / trades_per_day) if trades_per_day > 0 else 0
        
        # 3. Calculate Expectancy for Regimes
        def calc_regime_metrics(stats):
            w = stats["wins"]
            l = stats["losses"]
            n = stats["n_trades"]
            total_g = stats["total_gain"]
            total_l = stats["total_loss"]
            
            win_rate = (w / n * 100) if n > 0 else 0.0
            avg_win = (total_g / w) if w > 0 else 0.0
            avg_loss = (total_l / l) if l > 0 else 0.0
            
            # Expectancy
            win_pct = w / n if n > 0 else 0.0
            loss_pct = l / n if n > 0 else 0.0
            expectancy = (win_pct * avg_win) - (loss_pct * avg_loss)
            return win_rate, expectancy, n

        trend_wr, trend_exp, trend_n = calc_regime_metrics(regime_stats["Trend"])
        range_wr, range_exp, range_n = calc_regime_metrics(regime_stats["Range"])

        # Detailed Plotting for the Year
        print(f"Creating graph for {year}...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Metrics Math (Overall)
        initial_balance = env_instance.initial_balance
         # Fix: Remove the last point if it dropped to initial balance due to reset
        if len(history_net_worth) > 1 and history_net_worth[-1] == 10000 and history_net_worth[-2] != 10000:
             history_net_worth.pop()
        
        final_balance = history_net_worth[-1]
        total_profit = final_balance - initial_balance
        return_pct = (total_profit / initial_balance) * 100
        
        # Total Expectancy
        win_rate = (wins / n_trades * 100) if n_trades > 0 else 0.0
        avg_win = (total_gain / wins) if wins > 0 else 0.0
        avg_loss = (total_loss / losses) if losses > 0 else 0.0
        rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        profit_factor = (total_gain / total_loss) if total_loss > 0 else 0.0
        
        win_pct_decimal = wins / n_trades if n_trades > 0 else 0.0
        loss_pct_decimal = losses / n_trades if n_trades > 0 else 0.0
        expectancy = (win_pct_decimal * avg_win) - (loss_pct_decimal * avg_loss)

        print("-" * 33)
        print(f"--- {year} Professional Metrics ---")
        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Balance:   ${final_balance:.2f}")
        print(f"Total Return:    ${total_profit:.2f} ({return_pct:.2f}%)")
        print(f"Total Trades:    {n_trades}")
        print(f"Trade Frequency: {trades_per_day:.2f} trades/day (Avg ~{avg_time_between_trades_hours:.1f}h wait)")
        print(f"Win Rate:        {win_rate:.2f}%")
        print(f"Avg Win:         ${avg_win:.2f}")
        print(f"Avg Loss:        ${avg_loss:.2f}")
        print(f"Risk/Reward:     1:{rr_ratio:.2f}")
        print(f"Profit Factor:   {profit_factor:.2f}")
        print(f"Expectancy:      ${expectancy:.2f}")
        print(f"Max Drawdown:    {max_drawdown_pct:.2f}%")
        print(f"Max DD Duration: {max_duration_days:.1f} days ({max_duration_hours:.1f} hours)")
        print("-" * 33)
        print("--- Market Regime Analysis ---")
        print(f"[TREND (ADX>25)] Trades: {trend_n} | Win Rate: {trend_wr:.1f}% | Expectancy: ${trend_exp:.2f}")
        print(f"[RANGE (ADX<25)] Trades: {range_n} | Win Rate: {range_wr:.1f}% | Expectancy: ${range_exp:.2f}")
        print("-" * 33)

        if expectancy > 5 and profit_factor > 1.2: 
             print(">>> STATUS: PASS (Positive Expectancy)")
        else:
             print(">>> STATUS: FAIL (Negative Expectancy)")

        print("-" * 33)
        
        # Add Text to Plot
        stats_text = (
            f"Return: {return_pct:.2f}%\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Drawdown: {max_drawdown_pct:.2f}%\n"
            f"Longest DD: {max_duration_days:.1f}d\n"
            f"Trend Exp: ${trend_exp:.2f}\n"
            f"Range Exp: ${range_exp:.2f}"
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
        output_file = f"kaggle_output/backtest_result_{year}_FIXED.png" # Updated path to save in kaggle_output
        if not os.path.exists("kaggle_output"):
             os.makedirs("kaggle_output", exist_ok=True)
        plt.savefig(output_file)
        print(f"Saved {output_file}")
        plt.close()

if __name__ == "__main__":
    backtest()

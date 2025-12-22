import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """
    A custom stock trading environment for Gymnasium.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, commission_pct=0.0001, contract_size=1.0, leverage=100.0):
        super(StockTradingEnv, self).__init__()
        
        self.df_values = df.values
        self.df_columns = df.columns.tolist()
        try:
            self.close_idx = self.df_columns.index('Close')
            self.high_idx = self.df_columns.index('High')
            self.low_idx = self.df_columns.index('Low')
        except ValueError:
            raise ValueError("DataFrame must contain 'Close', 'High', and 'Low' columns")

        self.initial_balance = initial_balance
        self.commission_pct = commission_pct
        self.contract_size = contract_size 
        self.leverage = leverage 
        
        # RISK MANAGEMENT CONSTANTS
        self.STOP_LOSS_PCT = 0.01 # 1% (Safety)
        self.TAKE_PROFIT_PCT = 0.03 # 3% (Target 1:3 Risk:Reward)
        self.RISK_PER_TRADE = 0.02 # 2% Max Risk per trade
        
        # Action Space: 0=Hold, 1=Buy/Long, 2=Sell/Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space
        num_features = self.df_values.shape[1]
        self.obs_shape = num_features + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0 
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        self.initial_price = self.df_values[0][self.close_idx]
        self.entry_price = 0 
        self.realized_pnl = 0 
        return self._next_observation(), {}

    def _next_observation(self):
        market_data = self.df_values[self.current_step]
        # Normalize: Price / Initial (or relative)
        market_obs = market_data / self.initial_price
        
        # Position: 1 (Long), -1 (Short), 0 (Flat)
        if self.shares_held > 0:
            position_status = 1.0
        elif self.shares_held < 0:
            position_status = -1.0
        else:
            position_status = 0.0
            
        norm_balance = self.balance / self.initial_balance
        position_data = np.array([position_status, norm_balance], dtype=np.float32)
        
        obs = np.concatenate((market_obs, position_data))
        return obs.astype(np.float32)

    def step(self, action):
        self.realized_pnl = 0
        # 1. Check for Stop Loss / Take Profit / Margin Call
        terminated = False
        forced_close = False
        
        if self.shares_held != 0:
            current_low = self.df_values[self.current_step][self.low_idx]
            current_high = self.df_values[self.current_step][self.high_idx]
            
            # LONG CHECKS
            if self.shares_held > 0:
                # Stop Loss
                stop_price = self.entry_price * (1 - self.STOP_LOSS_PCT)
                # Take Profit (1:3 Target)
                tp_price = self.entry_price * (1 + self.TAKE_PROFIT_PCT)
                
                if current_low <= stop_price:
                    self._close_position(stop_price) 
                    forced_close = True
                elif current_high >= tp_price:
                    self._close_position(tp_price) # Hit 1:3 Target
                    forced_close = True
            
            # SHORT CHECKS
            elif self.shares_held < 0:
                # Stop Loss
                stop_price = self.entry_price * (1 + self.STOP_LOSS_PCT)
                # Take Profit (1:3 Target)
                tp_price = self.entry_price * (1 - self.TAKE_PROFIT_PCT)
                
                if current_high >= stop_price:
                    self._close_position(stop_price)
                    forced_close = True
                elif current_low <= tp_price:
                    self._close_position(tp_price) # Hit 1:3 Target
                    forced_close = True

        # 2. If not forced closed, take agent action
        if not forced_close:
            self._take_action(action)

        self.current_step += 1
        
        if self.current_step >= len(self.df_values):
            self.current_step = len(self.df_values) - 1
            terminated = True
        else:
            terminated = False
            
        current_price = self.df_values[self.current_step][self.close_idx]
        
        unrealized_pnl = 0
        if self.shares_held != 0:
            unrealized_pnl = (current_price - self.entry_price) * self.shares_held * self.contract_size

        new_net_worth = self.balance + unrealized_pnl
        
        reward = ((new_net_worth - self.net_worth) / self.net_worth) * 100
        
        self.net_worth = new_net_worth
        self.portfolio_history.append(self.net_worth)
        
        if self.net_worth <= 0:
            terminated = True
            reward = -100
            
        reward = np.clip(reward, -10, 10)
        
        return self._next_observation(), reward, terminated, False, {'pnl': self.realized_pnl}

    def _take_action(self, action):
        current_price = self.df_values[self.current_step][self.close_idx]
        
        # Action 0: Hold
        if action == 0: return

        # Action 1: Buy (Go Long)
        if action == 1: # Buy
            # If Short, Cover First
            if self.shares_held < 0: self._close_position(current_price)
            # If Flat, Open Long
            if self.shares_held == 0: self._open_position(current_price, side='long')

        # Action 2: Sell (Go Short)
        elif action == 2: # Sell
            # If Long, Sell/Close First
            if self.shares_held > 0: self._close_position(current_price)
            # If Flat, Open Short
            if self.shares_held == 0: self._open_position(current_price, side='short')

    def _open_position(self, price, side):
        # SMART SIZING
        # 1. Calc Risk Amount (e.g. 5% of Balance)
        risk_amount = self.balance * self.RISK_PER_TRADE
        
        # 2. Calc Stop Distance (e.g. 0.2% of Price)
        stop_distance = price * self.STOP_LOSS_PCT
        
        # 3. Calc Max Contracts allowed by Risk
        # Loss = Contracts * StopDist * ContractSize
        # Contracts = Risk / (StopDist * Size)
        # Avoid div by zero
        if stop_distance > 0:
             contracts_by_risk = int(risk_amount / (stop_distance * self.contract_size))
        else:
             contracts_by_risk = 0
             
        # 4. Calc Max Contracts allowed by Buying Power (Leverage)
        max_buying_power = self.balance * self.leverage
        contract_value = price * self.contract_size
        if contract_value > 0:
             contracts_by_leverage = int(max_buying_power / contract_value)
        else:
             contracts_by_leverage = 0
             
        # 5. Take the Minimum (Safer constrained size)
        num_contracts = min(contracts_by_risk, contracts_by_leverage)
        
        # Execute
        if num_contracts > 0:
            commission = num_contracts * contract_value * self.commission_pct
            
            # Check if we can afford commission
            if self.balance - commission < 0:
                # Reduce size to afford comm
                # Hard calculation, just reduce by 10% safety until it fits? 
                # Or simplify:
                num_contracts = int(num_contracts * 0.9)
                commission = num_contracts * contract_value * self.commission_pct
            
            if self.balance - commission > 0 and num_contracts > 0:
                self.balance -= commission
                self.entry_price = price
                if side == 'long':
                    self.shares_held = num_contracts
                else:
                    self.shares_held = -num_contracts

    def _close_position(self, price):
        # Realize PnL
        # PnL = (Exit - Entry) * Shares * Contract_Size
        pnl = (price - self.entry_price) * self.shares_held * self.contract_size
        
        contract_value = price * self.contract_size * abs(self.shares_held)
        commission = contract_value * self.commission_pct
        
        self.balance += (pnl - commission)
        self.shares_held = 0
        self.entry_price = 0
        self.realized_pnl = pnl - commission

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Shares: {self.shares_held}')

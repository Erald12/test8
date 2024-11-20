# Import necessary libraries
import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import ta
from scipy.stats import linregress
import neat
import time
import pickle
from datetime import datetime, timedelta


def sync_to_1_day():
    now = datetime.now()
    # Calculate the next midnight (next day at 00:00)
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    # Calculate the seconds until the next midnight
    seconds_left = (next_midnight - now).total_seconds()
    # Sleep until the next midnight
    if seconds_left > 0:
        time.sleep(seconds_left)


while True:
    try:
        # Initialize tvDatafeed
        tv = TvDatafeed()


        # Fetch data with retry logic
        def fetch_data_with_retry(tv, symbol, exchange, interval, n_bars, retries=3):
            """Fetch historical data with retry logic."""
            for attempt in range(retries):
                try:
                    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
                    if df is not None and not df.empty:
                        return df
                    else:
                        print(f"No data returned for {symbol}. Retrying... (Attempt {attempt + 1}/{retries})")
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}. Retrying... (Attempt {attempt + 1}/{retries})")
                time.sleep(2)  # Wait before retrying
            return None  # Return None if all retries fail


        # Specify parameters
        symbol_used = 'SOLUSDT.P'
        platform = 'OKX'
        n_bars = 6400

        # Fetch historical data for symbol_used
        df_symbolused = fetch_data_with_retry(tv, symbol_used, platform, Interval.in_5_minute, n_bars)

        if df_symbolused is None:
            raise ValueError("Failed to fetch data for symbol_used.")

        # Rename columns for consistency
        df_symbolused.columns = ['symbol', 'open_symbolused', 'high_symbolused', 'low_symbolused', 'close_symbolused',
                                 'volume_symbolused']

        # Symbols for additional data
        sol_symbol = 'SOL'
        btc_symbol = 'BTC'
        total_symbol = 'TOTAL'
        platform_cryptocap = 'CRYPTOCAP'


        # Function to calculate slope over a rolling window
        def calculate_slope(series, window=7):
            slopes = [np.nan] * (window - 1)  # Fill initial rows with NaN
            for i in range(window - 1, len(series)):
                y = series[i - window + 1:i + 1]
                x = np.arange(window)
                # Perform linear regression and extract the slope
                slope, _, _, _, _ = linregress(x, y)
                slopes.append(slope)
            return pd.Series(slopes, index=series.index)


        def calculate_rolling_std_high_low(high_series, low_series, window=7):
            """
            Calculate the rolling standard deviation using the average of high and low prices over a specified window.

            Parameters:
                high_series (pd.Series): The high prices series.
                low_series (pd.Series): The low prices series.
                window (int): The rolling window size.

            Returns:
                pd.Series: A series containing the rolling standard deviation of the average of high and low prices.
            """
            # Calculate the average of high and low prices
            avg_high_low = (high_series + low_series) / 2

            # Initialize a list with NaN for the initial values where the rolling window can't be applied
            rolling_std = [np.nan] * (window - 1)

            # Calculate the rolling standard deviation
            for i in range(window - 1, len(avg_high_low)):
                window_data = avg_high_low[i - window + 1:i + 1]
                std_dev = np.std(window_data)
                rolling_std.append(std_dev)
            return pd.Series(rolling_std, index=high_series.index)


        def calculate_rolling_correlation(series1, series2, window=7):
            """
            Calculate the rolling correlation between two price series over a specified window.

            Parameters:
                series1 (pd.Series): The first price series (e.g., BTC close prices).
                series2 (pd.Series): The second price series (e.g., ETH close prices).
                window (int): The rolling window size (default is 7).

            Returns:
                pd.Series: A series containing the rolling correlation values.
            """
            # Calculate the rolling correlation
            rolling_corr = series1.rolling(window).corr(series2)
            return rolling_corr


        # Fetch OHLC data for SOL ,BTC and TOTAL
        df_sol = fetch_data_with_retry(tv, sol_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)
        df_total = fetch_data_with_retry(tv, total_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)
        df_btc = fetch_data_with_retry(tv, btc_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)

        # Check if the data is fetched successfully
        if df_sol is None or df_total is None or df_btc is None:
            raise ValueError("Failed to fetch data for BTC or TOTAL.")

        # Rename columns for consistency
        df_sol.columns = ['symbol', 'open_sol', 'high_sol', 'low_sol', 'close_sol', 'volume_sol']
        df_total.columns = ['symbol', 'open_total', 'high_total', 'low_total', 'close_total', 'volume_total']
        df_btc.columns = ['symbol', 'open_btc', 'high_btc', 'low_btc', 'close_btc', 'volume_btc']

        # Merge the datasets on index (align the timestamps)
        df_combined = pd.concat([df_symbolused, df_sol, df_total, df_btc], axis=1)
        df_combined.dropna(inplace=True)  # Drop NaN values due to potential mismatched timestamps

        # Calculate Stochastic Slow and EMAs for the symbolused
        data = df_combined
        data['ema5'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=5).ema_indicator()
        data['ema8'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=8).ema_indicator()
        data['ema100'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=50).ema_indicator()

        # Calculate slopes for symbolused, btc, and total
        data['symbolused_slope'] = calculate_slope(data['close_symbolused'], window=7)
        data['sol_slope'] = calculate_slope(data['close_sol'], window=7)
        data['total_slope'] = calculate_slope(data['close_total'], window=7)
        data['btc_slope'] = calculate_slope(data['close_btc'], window=7)

        # Compute for Price ratio
        data['price_ratio_symbolused'] = (data['close_symbolused'] - data['open_symbolused']) / data['open_symbolused']
        data['price_ratio_sol'] = (data['close_sol'] - data['open_sol']) / data['open_sol']
        data['price_ratio_total'] = (data['close_total'] - data['open_total']) / data['open_total']
        data['price_ratio_btc'] = (data['close_btc'] - data['open_btc']) / data['open_btc']

        # Compute for RSI 14 for symbolused
        data['rsi14_symbolused'] = ta.momentum.RSIIndicator(close=data['close_symbolused'], window=14).rsi()

        # Compute for standard deviation
        data['std_symbolused'] = calculate_rolling_std_high_low(data['high_symbolused'], data['low_symbolused'])
        data['std_sol'] = calculate_rolling_std_high_low(data['high_sol'], data['low_sol'])
        data['std_total'] = calculate_rolling_std_high_low(data['high_total'], data['low_total'])
        data['std_btc'] = calculate_rolling_std_high_low(data['high_btc'], data['low_btc'])

        # Compute for correlation between sol and btc
        data['correlation_sol_btc'] = calculate_rolling_correlation(data['close_sol'], data['close_btc'])

        data.dropna(inplace=True)

        data_inputs = pd.DataFrame({
            'close_symbolused': data['close_symbolused'],
            'high_symbolused': data['high_symbolused'],
            'low_symbolused': data['low_symbolused'],
            'open_symbolused': data['open_symbolused'],
            'symbolused_slope': data['symbolused_slope'],
            'sol_slope': data['sol_slope'],
            'total_slope': data['total_slope'],
            'btc_slope': data['btc_slope'],
            'std_symbolused': data['std_symbolused'],
            'std_total': data['std_total'],
            'std_btc': data['std_btc'],
            'std_sol': data['std_sol'],
            'price_ratio_symbolused': data['price_ratio_symbolused'],
            'price_ratio_total': data['price_ratio_total'],
            'price_ratio_sol': data['price_ratio_sol'],
            'price_ratio_btc': data['price_ratio_btc'],
            'ema5': data['ema5'],
            'ema8': data['ema8'],
            'ema100': data['ema100'],
            'close_sol': data['close_sol'],
            'close_total': data['close_total'],
            'close_btc': data['close_btc'],
            'rsi14_symbolused': data['rsi14_symbolused'],
            'correlation_sol_btc': data['correlation_sol_btc']
        })

        # Split data into two halves for training and testing
        first_half = data_inputs[:len(data_inputs) - 2000]
        second_half = data_inputs[len(data_inputs) - 2000:]


        # Define a trading environment with a feedback loop
        class TradingEnvironment:
            def __init__(self, data, starting_balance=20):
                self.data = data
                self.balance = starting_balance
                self.current_step = 0
                self.winloss = []
                self.action_list = []
                self.equity_history = [starting_balance]
                self.profits = [float(starting_balance)]
                self.entry = []
                self.stop_loss = []
                self.take_profit = []
                self.leverage = []
                self.amount = []
                self.side = []
                self.close_price = data['close_symbolused'].values
                self.open_price = data['open_symbolused'].values
                self.high_price = data['high_symbolused'].values
                self.low_price = data['low_symbolused'].values
                self.ema5 = data['ema5'].values
                self.ema8 = data['ema8'].values
                self.ema100 = data['ema100'].values
                self.slope_symbolused = data['symbolused_slope'].values
                self.slope_total = data['total_slope'].values
                self.slope_btc = data['btc_slope'].values
                self.correlation_btc_sol = data['correlation_sol_btc'].values
                self.rsi14_symbolused = data['rsi14_symbolused'].values


            def reset(self):
                self.balance = 20
                self.current_step = 0
                self.winloss = []
                self.action_list = []
                self.equity_history = [self.balance]
                self.profits = [float(self.balance)]
                self.entry = []
                self.stop_loss = []
                self.take_profit = []
                self.leverage = []
                self.amount = []
                self.side = []

            def step(self, action):
                # Actions: 0 = hold, 1 = buy, -1 = sell
                price = self.close_price[self.current_step]

                ema5_current, ema8_current, ema100_current = self.ema5[self.current_step], self.ema8[self.current_step], \
                    self.ema100[self.current_step]
                ema5_prev, ema8_prev, ema100_prev = self.ema5[self.current_step - 1], self.ema8[self.current_step - 1], \
                    self.ema100[self.current_step - 1]

                step = 1

                if action == 1 and ema5_prev < ema8_prev and ema5_current > ema8_current and self.close_price[
                    self.current_step] > self.open_price[self.current_step] and self.close_price[
                    self.current_step] > ema100_current and self.close_price[
                    self.current_step - 1] < ema100_prev and len(self.entry) == 0:
                    loss_pct = 0.03
                    gain_pct = 0.15
                    stop_loss = self.low_price[self.current_step - 1]
                    leverage = abs(loss_pct / ((stop_loss - price) / price))
                    tp_limit = ((gain_pct * price) / abs(-loss_pct / ((stop_loss - price) / price))) + price
                    self.action_list.append(action)
                    self.entry.append(float(price))
                    self.stop_loss.append(float(stop_loss))
                    self.take_profit.append(float(tp_limit))
                    self.leverage.append(leverage)
                    self.side.append(1)
                    self.amount.append(sum(self.profits) - (sum(self.profits) * self.leverage[0] * 0.0005))

                elif action == 1 and ema5_prev > ema8_prev and ema5_current < ema8_current and self.close_price[
                    self.current_step] < self.open_price[self.current_step] and self.close_price[
                    self.current_step] < ema100_current and self.close_price[
                    self.current_step - 1] > ema100_prev and len(self.entry) == 0:
                    loss_pct = 0.03
                    gain_pct = 0.15
                    stop_loss = self.high_price[self.current_step - 1]
                    leverage = abs(loss_pct / ((stop_loss - price) / price))
                    tp_limit = ((-gain_pct * price) / abs(loss_pct / ((stop_loss - price) / price))) + price
                    self.leverage.append(leverage)
                    self.action_list.append(action)
                    self.entry.append(float(price))
                    self.stop_loss.append(float(stop_loss))
                    self.take_profit.append(float(tp_limit))
                    self.side.append(-1)
                    self.amount.append(sum(self.profits) - (sum(self.profits) * self.leverage[0] * 0.0005))

                elif action == -1 and len(self.entry) > 0:
                    self.action_list.append(-1)
                    if price > self.entry[0] and self.side[0] == 1:
                        self.winloss.append(1)
                        pct = abs((price - self.entry[0]) / self.entry[0])
                        self.profits.append((self.amount[0] * self.leverage[0] * pct) - (self.amount[0] * self.leverage[0] * 0.0005))
                        self.entry.clear()
                        self.take_profit.clear()
                        self.stop_loss.clear()
                        self.amount.clear()
                        self.leverage.clear()
                        self.side.clear()

                    elif price < self.entry[0] and self.side[0] == 1:
                        self.winloss.append(-1)
                        pct = abs((price - self.entry[0]) / self.entry[0])
                        self.profits.append(-(self.amount[0] * self.leverage[0] * pct) - (self.amount[0] * self.leverage[0] * 0.0005))
                        self.entry.clear()
                        self.take_profit.clear()
                        self.stop_loss.clear()
                        self.amount.clear()
                        self.leverage.clear()
                        self.side.clear()

                    elif price > self.entry[0] and self.side[0] == -1:
                        self.winloss.append(-1)
                        pct = abs((price - self.entry[0]) / self.entry[0])
                        self.profits.append(
                            -(self.amount[0] * self.leverage[0] * pct) - (self.amount[0] * self.leverage[0] * 0.0005))
                        self.entry.clear()
                        self.take_profit.clear()
                        self.stop_loss.clear()
                        self.amount.clear()
                        self.leverage.clear()
                        self.side.clear()

                    elif price < self.entry[0] and self.side[0] == -1:
                        self.winloss.append(1)
                        pct = abs((price - self.entry[0]) / self.entry[0])
                        self.profits.append(
                            (self.amount[0] * self.leverage[0] * pct) - (self.amount[0] * self.leverage[0] * 0.0005))
                        self.entry.clear()
                        self.take_profit.clear()
                        self.stop_loss.clear()
                        self.amount.clear()
                        self.leverage.clear()
                        self.side.clear()

                elif len(self.side) > 0:
                    self.action_list.append(0)

                    if self.side[0] == 1 and self.low_price[self.current_step] <= self.stop_loss[0]:
                        self.balance -= self.balance * 0.03
                        self.winloss.append(-1)
                        self.profits.append(-(self.amount[0] * 0.03) - (self.amount[0] * self.leverage[0] * 0.0005))
                        self.entry.clear()
                        self.take_profit.clear()
                        self.stop_loss.clear()
                        self.amount.clear()
                        self.leverage.clear()
                        self.side.clear()

                    elif self.side[0] == -1 and self.high_price[self.current_step] >= self.stop_loss[0]:
                        self.balance -= self.balance * 0.03
                        self.winloss.append(-1)
                        self.profits.append(-(self.amount[0] * 0.03) - (self.amount[0] * self.leverage[0] * 0.0005))
                        self.entry.clear()
                        self.take_profit.clear()
                        self.stop_loss.clear()
                        self.amount.clear()
                        self.leverage.clear()
                        self.side.clear()

                else:
                    self.action_list.append(0)

                self.current_step += step
                self.equity_history.append(self.balance)
                done = self.current_step >= len(self.data) - 1
                return self.balance, done, self.winloss, sum(self.profits), self.action_list


        def run_neat(config_path, save_path="best_genome3.pkl", generations=999999999999):
            """
            Runs the NEAT algorithm with a fitness function and saves the best genome.

            Args:
                config_path (str): Path to the NEAT configuration file.
                save_path (str): Path to save the best genome and its network.
                generations (int): Number of generations to run.
                winrate_weight (float): Weight of winrate in the fitness score.
                pnl_weight (float): Weight of profit and loss (PNL) in the fitness score.
            """
            # Load NEAT configuration
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)

            # Create a population
            p = neat.Population(config)

            # Add reporters to monitor progress
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)

            # Define fitness function for NEAT
            def evaluate_genomes(genomes, config):
                nonlocal best_genome, best_net  # Use outer-scope variables
                best_fitness = float('-inf')
                best_test = float('-inf')

                saved_best_fitness = 0
                saved_test_fitness = 0
                for genome_id, genome in genomes:
                    genome.fitness = 0.0  # Initialize fitness

                    # try:
                    # Create the neural network for the genome
                    net = neat.nn.RecurrentNetwork.create(genome, config)
                    env = TradingEnvironment(first_half)  # Assumes `data_inputs` is defined globally or passed in
                    env.reset()

                    env2 = TradingEnvironment(second_half)  # Assumes `data_inputs` is defined globally or passed in
                    env2.reset()

                    total_profit = 0
                    while True:
                        def min_max_scale():
                            distance1 = 0
                            distance2 = 0
                            if len(env.side)>0:
                                if env.side[0] == 1:
                                    if env.low_price[env.current_step] <= env.stop_loss[0] and env.high_price[env.current_step] > env.stop_loss[0]:
                                        distance1 = 0
                                        distance2 = (env.high_price[env.current_step] - env.stop_loss[0]) / (env.take_profit[0] - env.stop_loss[0])
                                    elif env.low_price[env.current_step] <= env.stop_loss[0] and env.high_price[env.current_step] <= env.stop_loss[0]:
                                        distance1 = 0
                                        distance2 = 0
                                    elif env.high_price[env.current_step] >= env.take_profit[0] and env.low_price[env.current_step] < env.take_profit[0]:
                                        distance1 = (env.low_price[env.current_step] - env.stop_loss[0]) / (env.take_profit[0] - env.stop_loss[0])
                                        distance2 = 1
                                    elif env.high_price[env.current_step] >= env.take_profit[0] and env.low_price[env.current_step] >= env.take_profit[0]:
                                        distance1 = 1
                                        distance2 = 1

                            else:
                                distance1 = 0
                                distance2 = 0
                            return distance1, distance2

                        def min_max_scale1():
                            distance1 = 0
                            distance2 = 0
                            if len(env.side) > 0:
                                if env.side[0] == -1:
                                    if env.high_price[env.current_step] >= env.stop_loss[0] and env.low_price[env.current_step] < env.stop_loss[0]:
                                        distance1 = (env.low_price[env.current_step] - env.take_profit[0]) / (env.stop_loss[0] - env.take_profit[0])
                                        distance2 = 1
                                    elif env.high_price[env.current_step] >= env.stop_loss[0] and env.low_price[env.current_step] >= env.stop_loss[0]:
                                        distance1 = 1
                                        distance2 = 1
                                    elif env.low_price[env.current_step] <= env.take_profit[0] and env.high_price[env.current_step] > env.take_profit[0]:
                                        distance1 = 0
                                        distance2 = (env.high_price[env.current_step] - env.take_profit[0]) / (env.stop_loss[0] - env.take_profit[0])
                                    elif env.low_price[env.current_step] <= env.take_profit[0] and env.high_price[env.current_step] <= env.take_profit[0]:
                                        distance1 = 0
                                        distance2 = 0
                            else:
                                distance1 = 0
                                distance2 = 0
                            return distance1, distance2

                        def signal():
                            if env.ema5[env.current_step -1] < env.ema8[env.current_step -1] and env.ema5[env.current_step] > env.ema8[env.current_step] and env.close_price[env.current_step] > env.open_price[env.current_step] and env.close_price[env.current_step] > env.ema100[env.current_step] and env.close_price[env.current_step - 1] < env.ema100[env.current_step-1]:
                                signal = 1
                            elif env.ema5[env.current_step -1] > env.ema8[env.current_step -1] and env.ema5[env.current_step] < env.ema8[env.current_step] and env.close_price[env.current_step] < env.open_price[env.current_step] and env.close_price[env.current_step] < env.ema100[env.current_step] and env.close_price[env.current_step - 1] > env.ema100[env.current_step-1]:
                                signal = -1
                            else:
                                signal = 0
                            return signal

                        distance1, distance2 = min_max_scale()
                        distance11, distance12 = min_max_scale1()
                        # Ensure state is compatible with neural network input size
                        state = np.concatenate([[env.slope_symbolused[env.current_step]], [env.rsi14_symbolused[env.current_step]], [distance1], [distance2],[distance11],[distance12], [signal()]])
                        action = np.argmax(net.activate(state)) - 1  # Map to -1, 0, 1

                        # Execute the action in the environment
                        balance, done, winloss, total_profits, action_list = env.step(action)
                        if done:
                            break


                    while True:
                        def min_max_scale2():
                            distance1 = 0
                            distance2 = 0
                            if len(env2.side)>0:
                                if env2.side[0] == 1:
                                    if env2.low_price[env2.current_step] <= env2.stop_loss[0] and env2.high_price[env2.current_step] > env2.stop_loss[0]:
                                        distance1 = 0
                                        distance2 = (env2.high_price[env2.current_step] - env2.stop_loss[0]) / (env2.take_profit[0] - env2.stop_loss[0])
                                    elif env2.low_price[env2.current_step] <= env2.stop_loss[0] and env2.high_price[env2.current_step] <= env2.stop_loss[0]:
                                        distance1 = 0
                                        distance2 = 0
                                    elif env2.high_price[env2.current_step] >= env2.take_profit[0] and env2.low_price[env2.current_step] < env2.take_profit[0]:
                                        distance1 = (env2.low_price[env2.current_step] - env2.stop_loss[0]) / (env2.take_profit[0] - env2.stop_loss[0])
                                        distance2 = 1
                                    elif env2.high_price[env2.current_step] >= env2.take_profit[0] and env2.low_price[env2.current_step] >= env2.take_profit[0]:
                                        distance1 = 1
                                        distance2 = 1
                            else:
                                distance1 = 0
                                distance2 = 0
                            return distance1, distance2

                        def min_max_scale21():
                            distance1 = 0
                            distance2 = 0
                            if len(env2.side)>0:
                                if env2.side[0] == -1:
                                    if env2.high_price[env2.current_step] >= env2.stop_loss[0] and env2.low_price[env2.current_step] < env2.stop_loss[0]:
                                        distance1 = (env2.low_price[env2.current_step] - env2.take_profit[0]) / (env2.stop_loss[0] - env2.take_profit[0])
                                        distance2 = 1
                                    elif env2.high_price[env2.current_step] >= env2.stop_loss[0] and env2.low_price[env2.current_step] >= env2.stop_loss[0]:
                                        distance1 = 1
                                        distance2 = 1
                                    elif env2.low_price[env2.current_step] <= env2.take_profit[0] and env2.high_price[env2.current_step] > env2.take_profit[0]:
                                        distance1 = 0
                                        distance2 = (env2.high_price[env2.current_step] - env2.take_profit[0]) / (env2.stop_loss[0] - env2.take_profit[0])
                                    elif env2.low_price[env2.current_step] <= env2.take_profit[0] and env2.high_price[env2.current_step] <= env2.take_profit[0]:
                                        distance1 = 0
                                        distance2 = 0
                            else:
                                distance1 = 0
                                distance2 = 0
                            return distance1, distance2

                        def signal2():
                            if env2.ema5[env2.current_step -1] < env2.ema8[env2.current_step -1] and env2.ema5[env2.current_step] > env2.ema8[env2.current_step] and env2.close_price[env2.current_step] > env2.open_price[env2.current_step] and env2.close_price[env2.current_step] > env2.ema100[env2.current_step] and env2.close_price[env2.current_step - 1] < env2.ema100[env2.current_step-1]:
                                signal = 1
                            elif env2.ema5[env2.current_step -1] > env2.ema8[env2.current_step -1] and env2.ema5[env2.current_step] < env2.ema8[env2.current_step] and env2.close_price[env2.current_step] < env2.open_price[env2.current_step] and env2.close_price[env2.current_step] < env2.ema100[env2.current_step] and env2.close_price[env2.current_step - 1] > env2.ema100[env2.current_step-1]:
                                signal = -1
                            else:
                                signal = 0
                            return signal

                        distance21, distance22 = min_max_scale2()
                        distance31, distance32 = min_max_scale21()
                        # Ensure state is compatible with neural network input size
                        state2 = np.concatenate([[env2.slope_symbolused[env2.current_step]], [env2.rsi14_symbolused[env2.current_step]], [distance21], [distance22],[distance31],[distance32], [signal2()]])
                        action2 = np.argmax(net.activate(state2)) - 1  # Map to -1, 0, 1

                        # Execute the action in the environment
                        balance, done, winloss, total_profits2, action_list = env2.step(action2)
                        if done:
                            break

                    profit = balance
                    total_profit = total_profits
                    win = winloss.count(1)
                    loss = winloss.count(-1)
                    PNL = (total_profit - 20) * 100 / 20
                    PNL2 = (total_profits2 - 20) * 100 / 20
                    winrate = win / len(winloss) if len(winloss) > 0 else 0

                    if total_profits>20 and total_profits2<20:
                        genome.fitness = 0
                    else:
                        genome.fitness = max(total_profit, 0)

                    if total_profits2<best_test:
                        genome.fitness = 0
                    else:
                        genome.fitness = max(total_profit, 0)

                    # Print genome performance
                    print(f'Trader: {genome_id}, PNL%: {round(PNL, 2)}%, PNL2%: {round(PNL2, 2)}%')

                    # Track the best genome
                    if genome.fitness > best_fitness:
                        best_fitness = genome.fitness
                        best_test = total_profits2
                        best_genome = genome
                        best_net = net

                    # except Exception as e:
                    #    print(f"Error evaluating genome {genome_id}: {e}")
                    #    genome.fitness = 0  # Avoid NoneType fitness

                # Save the best genome and network after evaluations
                if best_genome and best_fitness > 30 and best_test > 30:
                    saved_best_fitness = best_fitness
                    saved_test_fitness = best_test
                    with open(save_path, 'wb') as f:
                        pickle.dump({'genome': best_genome, 'network': best_net}, f)
                    print(f"Saved the best genome and network to {save_path}")
                print(f'Saved Best Fitness: {saved_best_fitness}, Saved Test Fitness: {saved_test_fitness}')

            # Initialize variables for the best genome and network
            best_genome = None
            best_net = None

            # Run the NEAT algorithm
            p.run(evaluate_genomes, generations)



            print('\nBest genome:\n{!s}'.format(best_genome))


        def create_neat_config():
            config_content = """
            [NEAT]
            pop_size = 100
            fitness_criterion = max
            fitness_threshold = 999999999999999999999
            reset_on_extinction = True

            [DefaultGenome]
            feed_forward = False

            # Node activation functions
            activation_default = sigmoid
            activation_mutate_rate = 0.0
            activation_options = sigmoid softmax hard_sigmoid scaled_sigmoid logistic tanh relu

            # Node aggregation functions
            aggregation_default = sum
            aggregation_mutate_rate = 0.0
            aggregation_options = sum mean product

            # Structural mutation rates
            single_structural_mutation = False
            structural_mutation_surer = 0
            conn_add_prob = 0.5
            conn_delete_prob = 0.2
            node_add_prob = 0.2
            node_delete_prob = 0.2

            # Connection parameters
            initial_connection = full_direct
            bias_init_mean = 0.0
            bias_init_stdev = 1.0
            bias_max_value = 30.0
            bias_min_value = -30.0
            bias_mutate_power = 0.5
            bias_mutate_rate = 0.1
            bias_replace_rate = 0.1

            # Response parameters (added these)
            response_init_mean = 0.0
            response_init_stdev = 1.0
            response_replace_rate = 0.1
            response_mutate_rate = 0.1
            response_mutate_power = 0.5
            response_max_value = 30.0
            response_min_value = -30.0

            # Default enabled state
            enabled_default = True

            # Enable mutation rate
            enabled_mutate_rate = 0.1

            # Node parameters
            num_hidden = 0
            num_inputs = 7
            num_outputs = 3

            # Connection mutation
            weight_init_mean = 0.0
            weight_init_stdev = 1.0
            weight_max_value = 30
            weight_min_value = -30
            weight_mutate_power = 0.5
            weight_mutate_rate = 0.8
            weight_replace_rate = 0.1

            # Compatibility parameters
            compatibility_disjoint_coefficient = 1.0
            compatibility_weight_coefficient = 0.5

            [DefaultSpeciesSet]
            compatibility_threshold = 3.0

            [DefaultStagnation]
            species_fitness_func = max
            max_stagnation = 15
            species_elitism = 2

            [DefaultReproduction]
            elitism = 2
            survival_threshold = 0.2
            """
            with open('neat_config6.txt', 'w') as f:
                f.write(config_content)


        # Function to test the trained NEAT model on the test data
        def test_model(net, test_data, config_path):
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
            # net = neat.nn.FeedForwardNetwork.create(genome, config)  # Create the neural network from the best genome
            env = TradingEnvironment(test_data)  # Initialize the environment with test data
            env.reset()  # Reset the environment

            total_profit = 0  # Variable to accumulate profit during testing
            while True:
                state = env.data.iloc[env.current_step, :24].values  # Get the current state from the environment
                action = np.argmax(net.activate(state)) - 1  # Choose action from the neural network
                balance, done, winloss, total_profit, action_list = env.step(
                    action)  # Step the environment with the chosen action
                if done:  # Break the loop if the episode is done
                    break

            total_profits = total_profit  # Calculate profit (final balance - initial balance)
            return total_profits


        def load_best_genome(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                winner_genome = data['genome']
                net = data['network']  # This is the loaded network, fully instantiated
            return winner_genome, net


        if __name__ == "__main__":
            # Provide path to your NEAT config file
            # Create configuration file
            create_neat_config()
            config_path = "neat_config6.txt"
            run_neat(config_path)
            best_genome, network = load_best_genome('best_genome3.pkl')
            test = test_model(network, data_inputs, config_path)
            print(f'total balance: {test}')
            sync_to_1_day()
    except Exception as e:
        print(e)





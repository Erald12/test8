# Import necessary libraries
import ta
from tvDatafeed import TvDatafeed, Interval
import numpy as np
import pandas as pd
import time
from scipy.stats import linregress
import neat
import pickle
from rich.console import Console
from datetime import datetime, timedelta
from datetime import datetime


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
                self.profits.append(
                    (self.amount[0] * self.leverage[0] * pct) - (self.amount[0] * self.leverage[0] * 0.0005))
                self.entry.clear()
                self.take_profit.clear()
                self.stop_loss.clear()
                self.amount.clear()
                self.leverage.clear()
                self.side.clear()

            elif price < self.entry[0] and self.side[0] == 1:
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
        done = self.current_step >= len(self.data)
        return self.balance, done, self.winloss, sum(self.profits), self.action_list


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
    with open('neat_config5.txt', 'w') as f:
        f.write(config_content)


# Function to test the trained NEAT model on the test data
def test_model(network, test_data):
    net = network
    env = TradingEnvironment(test_data)  # Initialize the environment with test data
    env.reset()  # Reset the environment

    actions = []

    while True:
        def min_max_scale():
            distance1 = 0
            distance2 = 0
            if len(env.side) > 0:
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
            if env.ema5[env.current_step - 1] < env.ema8[env.current_step - 1] and env.ema5[env.current_step] > env.ema8[env.current_step] and env.close_price[env.current_step] > env.open_price[env.current_step] and env.close_price[env.current_step] > env.ema100[env.current_step] and env.close_price[env.current_step - 1] < env.ema100[env.current_step - 1]:
                signal = 1
            elif env.ema5[env.current_step - 1] > env.ema8[env.current_step - 1] and env.ema5[env.current_step] < env.ema8[env.current_step] and env.close_price[env.current_step] < env.open_price[env.current_step] and env.close_price[env.current_step] < env.ema100[env.current_step] and env.close_price[env.current_step - 1] > env.ema100[env.current_step - 1]:
                signal = -1
            else:
                signal = 0
            return signal

        distance1, distance2 = min_max_scale()
        distance11, distance12 = min_max_scale1()
        # Ensure state is compatible with neural network input size
        state = np.concatenate([[env.slope_symbolused[env.current_step]], [env.rsi14_symbolused[env.current_step]], [distance1],[distance2], [distance11], [distance12], [signal()]])
        action = np.argmax(net.activate(state)) - 1  # Map to -1, 0, 1

        # Execute the action in the environment
        balance, done, winloss, total_profits, action_list = env.step(action)
        if done:
            break
    set = winloss
    win = set.count(1)
    loss = set.count(-1)
    if len(set) == 0:
        winrate = 0
    else:
        winrate = set.count(1) / len(set)
    PNL = (total_profits-20)*100/20
    print(f'PNL%: {round(PNL,2)}%')
    return action_list


def load_best_genome(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        winner_genome = data['genome']
        net = data['network']  # This is the loaded network, fully instantiated
    return winner_genome, net

def compute_trade_details(entry_price, sl_price, amount_usdt, position_type):
    loss_pct = 0.03
    gain_pct = 0.15
    if position_type == "short":
        distance = (sl_price - entry_price) / entry_price
        distance2 = loss_pct / (distance)
        leverage = abs(distance2)
        tp_price = ((-gain_pct * entry_price) / (leverage)) + entry_price
        position_size = amount_usdt * leverage
    elif position_type == "long":
        distance = (sl_price - entry_price) / entry_price
        distance2 = -loss_pct / (distance)
        leverage = abs(distance2)
        tp_price = ((gain_pct * entry_price) / (leverage)) + entry_price
        position_size = amount_usdt * leverage

    return position_size, round(tp_price,2), round(leverage,2)

#Set Leverage Function
def set_leverage(symbol,leverage, marginmode):
    result = accountAPI.set_leverage(
        instId=symbol,
        lever=str(leverage),
        mgnMode=marginmode
    )
    return result

# Function to get the latest price of SOL-USDT-SWAP
def get_latest_price(symbol):
    try:
        # Fetch the ticker information for the specified symbol
        ticker = okx.fetch_ticker(symbol)
        # Extract the last price from the ticker information
        last_price = ticker['last']
        return last_price
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None

def place_futures_order(symbol, side, size, trigger_price):
    try:
        # Place an algo order with trigger price
        result = tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode="isolated",  # Isolated margin mode
            side=side,  # "buy" for long, "sell" for short
            ordType="trigger",  # Trigger order type
            sz=str(size),  # Number of contracts
            triggerPx=str(trigger_price),  # Trigger price for the order to be activated
            orderPx=-1,
            triggerPxType="mark",
        )
        return result
    except Exception as e:
        print(f"Error placing buy order: {e}")
        return None

def place_stop_loss_in_buy(symbol,sl_price,order_size,marginmode):
    try:
        # Place a stop-loss order (sell) with the same size as the buy order
        result = tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode=marginmode,  # Isolated margin mode
            side="sell",  # Stop-loss should be on the sell side (to close the long position)
            ordType="conditional",  # Trigger order type
            sz=str(order_size),  # Use the same size as the recent buy order
            slTriggerPx=str(sl_price),  # Stop-loss trigger price
            slOrdPx=-1,  # Stop-loss execution price (lower than trigger price)
            slTriggerPxType="mark"  # Use the mark price as reference
        )

        # Print result (order details)
        return result

    except Exception as e:
        print(f"Error placing stop-loss order: {e}")

def place_stop_loss_in_sell(symbol, sl_price, order_size, marginmode):
    try:
        # Place a stop-loss order (sell) with the same size as the buy order
        result = tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode=marginmode,  # Isolated margin mode
            side="buy",  # Stop-loss should be on the sell side (to close the long position)
            ordType="conditional",  # Trigger order type
            sz=str(order_size),  # Use the same size as the recent buy order
            slTriggerPx=str(sl_price),  # Stop-loss trigger price
            slOrdPx=-1,  # Stop-loss execution price (lower than trigger price)
            slTriggerPxType="mark"  # Use the mark price as reference
        )
        return result

    except Exception as e:
        print(f"Error placing stop-loss order: {e}")

def place_tp_in_buy(symbol, tp_price, order_size, marginmode):
    try:
        # Place a stop-loss order (sell) with the same size as the buy order
        result = tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode=marginmode,  # Isolated margin mode
            side="sell",  # Stop-loss should be on the sell side (to close the long position)
            ordType="conditional",  # Trigger order type
            sz=str(order_size),  # Use the same size as the recent buy order
            tpTriggerPx=str(tp_price),  # Stop-loss trigger price
            tpOrdPx=-1,  # Stop-loss execution price (lower than trigger price)
            tpTriggerPxType="mark"  # Use the mark price as reference
        )

        # Print result (order details)
        return result

    except Exception as e:
        print(f"Error placing stop-loss order: {e}")

def place_tp_in_sell(symbol, tp_price, order_size, marginmode):
    try:
        # Place a stop-loss order (sell) with the same size as the buy order
        result = tradeAPI.place_algo_order(
            instId=symbol,  # Instrument ID
            tdMode=marginmode,  # Isolated margin mode
            side="buy",  # Stop-loss should be on the sell side (to close the long position)
            ordType="conditional",  # Trigger order type
            sz=str(order_size),  # Use the same size as the recent buy order
            tpTriggerPx=str(tp_price),  # Stop-loss trigger price
            tpOrdPx=-1,  # Stop-loss execution price (lower than trigger price)
            tpTriggerPxType="mark"  # Use the mark price as reference
        )

        # Print result (order details)
        return result

    except Exception as e:
        print(f"Error placing stop-loss order: {e}")

def sync_to_5_minute():
    now = datetime.now()
    # Calculate the next 5-minute mark
    next_five_minute = (now + timedelta(minutes=5 - (now.minute % 5))).replace(second=0, microsecond=0)
    # Calculate the seconds until the next 5-minute mark
    seconds_left = (next_five_minute - now).total_seconds()
    # Sleep until the next 5-minute mark
    if seconds_left > 0:
        time.sleep(seconds_left)


if __name__ == "__main__":
    # Provide path to your NEAT config file

    entry_prices = []
    sl_price_set = []
    tp_price_set = []
    side = []
    order_alg_id = []
    stop_alg_id = []
    tp_alg_id = []

    symbol_2 = input('Enter your TICKER in OKX for trading: (eg. SOL-USDT-SWAP, BTC-USDT-SWAP)')

    #Go Live Trading
    while True:
        try:
            logic = [True]
            while logic[-1]:
                try:
                    # Create configuration file
                    create_neat_config()
                    config_path = "neat_config6.txt"

                    # Load the best genome for testing
                    path_to_best_genome = 'best_genome3.pkl'  # Path to the saved genome file
                    best_genome, net = load_best_genome(path_to_best_genome)
                    # Initialize tvDatafeed
                    tv = TvDatafeed()

                    # Specify parameters
                    symbol_used = 'SOLUSDT.P'
                    platform = 'OKX'
                    n_bars = 5000

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
                    logic.append(False)
                except Exception as e:
                    print(e)
                    logic.append(True)

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
            data['price_ratio_symbolused'] = (data['close_symbolused'] - data['open_symbolused']) / data[
                'open_symbolused']
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

            ema5 = data_inputs['ema5']
            ema8 = data_inputs['ema8']
            ema100 = data_inputs['ema100']
            close_symbolused = data_inputs['close_symbolused']
            open_symbolused = data_inputs['open_symbolused']
            high_symbolused = data_inputs['high_symbolused']
            low_symbolused = data_inputs['low_symbolused']

            actions = test_model(net, data_inputs)

            import ccxt
            import okx.Trade as Trade
            import okx.Account as Account

            # API credentials
            apikey = "e5f90b15-e3f1-469d-9477-11fdfcf04fdb"
            secretkey = "F79E3CE531674965F95CAEDEA3C81C69"
            passphrase = "Easy09159562534*"

            marginmode = "isolated"

            # Initialize OKX exchange with your API credentials
            okx = ccxt.okx({
                'apiKey': apikey,
                'secret': secretkey,
                'password': passphrase,
                'enableRateLimit': True,  # Ensures you don't exceed rate limits
            })

            flag = "0"  # Production trading: 0, Demo trading: 1

            # Initialize TradeAPI and AccountAPI
            tradeAPI = Trade.TradeAPI(apikey, secretkey, passphrase, False, flag)
            accountAPI = Account.AccountAPI(apikey, secretkey, passphrase, False, flag)

            # Fetch your overall balance
            balance = okx.fetch_balance()

            # Access USDT balance
            usdt_balance = balance['total'].get('USDT', 0)


            console = Console()
            positions_recent = okx.fetch_positions()
            current_price = get_latest_price(symbol_2)

            if len(positions_recent) == 0:
                entry_prices.clear()
                sl_price_set.clear()
                tp_price_set.clear()
                side.clear()
                order_alg_id.clear()
                stop_alg_id.clear()
                tp_alg_id.clear()
                print(f'Price Entered: {entry_prices}, Stop Loss: {sl_price_set}, Side: {side}')
                # Execute the action (you can integrate with a broker API to execute buy/sell)
                if data_inputs['ema5'].iloc[-3]<data_inputs['ema8'].iloc[-3] and data_inputs['ema5'].iloc[-2] > data_inputs['ema8'].iloc[-2] and data_inputs['close_symbolused'].iloc[-2]>data_inputs['open_symbolused'].iloc[-2] and data_inputs['close_symbolused'].iloc[-2]>data_inputs['ema100'].iloc[-2] and data_inputs['close_symbolused'].iloc[-3]<data_inputs['ema100'].iloc[-3]:  # Buy action
                    console.print('[yellow]Bullish Crossover Detected[/yellow]')

                    # Bullish crossover: Buy when short EMA > long EMA
                    if actions[-2] == 1:
                        console.print('[yellow]Predictions Confirmed[/yellow]')
                        entry_price = data_inputs['close_symbolused'].iloc[-2]
                        entry_prices.append(entry_price)
                        sl_price = data_inputs['low_symbolused'].iloc[-3]
                        sl_price_set.append(sl_price)
                        position_size_usdt, take_profit_price, leverage = compute_trade_details(entry_price, sl_price,usdt_balance, "long")
                        side.append(1)
                        tp_price_set.append(take_profit_price)
                        console.print(f'[yellow]Action: Buy (Bullish EMA Crossover). SL: {sl_price}. Entry: {entry_price}. TP: {take_profit_price}, Leverage: {leverage}, Position Size: {position_size_usdt}[/yellow]')
                        size = position_size_usdt / get_latest_price(symbol_2)
                        set_leverage(symbol_2, leverage, marginmode)
                        # Place a buy order with your broker API
                        if leverage < 50:
                            order = place_futures_order(symbol_2, "buy", round(size, 2), entry_price)
                            print(order)

                            # Extract algoId from the order response
                            order_algo_id = order['data'][0]['algoId']
                            order_alg_id.append(order_algo_id)

                            # Wait until the order is triggered
                            while True:
                                try:
                                    positions = okx.fetch_positions()
                                    current_price = get_latest_price(symbol_2)

                                    # Check if the order is filled
                                    if len(positions) > 0:  # Adjusted to check for any filled positions
                                        for position in positions:
                                            position_size = position['contracts']
                                            positions = okx.fetch_positions()

                                            # Place stop-loss and take-profit orders
                                            stop_loss_order = place_stop_loss_in_buy(symbol_2, sl_price, position_size, marginmode)
                                            stop_loss_algo_id = stop_loss_order['data'][0]['algoId']
                                            stop_alg_id.append(stop_loss_algo_id)
                                            print(f'Stop_Loss Order: {stop_loss_order}')
                                        # Exit the position management loop after handling the position
                                        break
                                    elif len(positions) == 0 and current_price <= sl_price or current_price >= take_profit_price:
                                        algo_orders = [{"instId": symbol_2, "algoId": order_algo_id}]
                                        result = tradeAPI.cancel_algo_order(algo_orders)
                                        entry_prices.clear()
                                        sl_price_set.clear()
                                        tp_price_set.clear()
                                        side.clear()
                                        order_alg_id.clear()
                                        stop_alg_id.clear()
                                        tp_alg_id.clear()
                                        break
                                except Exception as e:
                                    print(e)
                        else:
                            console.print('[yellow]Leverage is too high, order Cancelled.[/yellow]')

                elif data_inputs['ema5'].iloc[-3]>data_inputs['ema8'].iloc[-3] and data_inputs['ema5'].iloc[-2]<data_inputs['ema8'].iloc[-2] and data_inputs['close_symbolused'].iloc[-2]<data_inputs['open_symbolused'].iloc[-2] and data_inputs['close_symbolused'].iloc[-2]<data_inputs['ema100'].iloc[-2] and data_inputs['close_symbolused'].iloc[-3]>data_inputs['ema100'].iloc[-3]:  # Sell action
                    console.print('[yellow]Bearish Crossover Detected[/yellow]')

                    # Bearish crossover: Sell when short EMA < long EMA
                    if actions[-2] == 1:
                        console.print('[yellow]Predictions Confirmed[/yellow]')
                        entry_price = data_inputs['close_symbolused'].iloc[-2]
                        entry_prices.append(entry_price)
                        sl_price = data_inputs['high_symbolused'].iloc[-3]
                        sl_price_set.append(sl_price)
                        position_size_usdt, take_profit_price, leverage = compute_trade_details(entry_price, sl_price,usdt_balance, "short")
                        tp_price_set.append(take_profit_price)
                        side.append(-1)
                        size = position_size_usdt / get_latest_price(symbol_2)
                        set_leverage(symbol_2, leverage, marginmode)
                        console.print(f'[yellow]Action: Sell (Bearish EMA Crossover). SL: {sl_price}. Entry: {entry_price}. TP: {take_profit_price}, Leverage: {leverage}, Position Size: {position_size_usdt}[/yellow]')

                        if leverage < 50:
                            # Place a sell order with your broker API
                            order = place_futures_order(symbol_2, "sell", round(size, 2), entry_price)
                            print(order)

                            # Extract algoId from the order response
                            order_algo_id = order['data'][0]['algoId']
                            order_alg_id.append(order_algo_id)

                            # Wait until the order is triggered
                            while True:
                                try:
                                    positions = okx.fetch_positions()
                                    current_price = get_latest_price(symbol_2)
                                    # Check if the order is filled
                                    if len(positions) > 0:  # Adjusted to check for any filled positions
                                        for position in positions:
                                            position_size = position['contracts']
                                            current_price = get_latest_price(symbol_2)

                                            # Place stop-loss and take-profit orders
                                            stop_loss_order = place_stop_loss_in_sell(symbol_2, sl_price, position_size, marginmode)
                                            stop_loss_algo_id = stop_loss_order['data'][0]['algoId']
                                            stop_alg_id.append(stop_loss_algo_id)
                                            print(f'Stop_Loss Order: {stop_loss_order}')
                                        # Exit the position management loop after handling the position
                                        break

                                    elif len(positions) == 0 and current_price >= sl_price or current_price <= take_profit_price:
                                        algo_orders = [{"instId": symbol_2, "algoId": order_algo_id}]
                                        result = tradeAPI.cancel_algo_order(algo_orders)
                                        entry_prices.clear()
                                        sl_price_set.clear()
                                        tp_price_set.clear()
                                        side.clear()
                                        order_alg_id.clear()
                                        stop_alg_id.clear()
                                        tp_alg_id.clear()
                                        break
                                except Exception as e:
                                    print(e)
                        else:
                            console.print('[yellow]Leverage is too high, order cancelled[/yellow]')
                else:
                    prev_price = data_inputs['close_symbolused'].iloc[-2]
                    prev_open = data_inputs['open_symbolused'].iloc[-2]
                    prev_high = data_inputs['high_symbolused'].iloc[-2]
                    prev_low = data_inputs['low_symbolused'].iloc[-2]
                    ema5 = data_inputs['ema5'].iloc[-2]
                    ema8 = data_inputs['ema8'].iloc[-2]
                    ema100 = data_inputs['ema100'].iloc[-2]
                    console.print(f'[green]Action: Hold. High: {prev_high}. Low: {prev_low}. Open: {prev_open}. Close: {prev_price}. EMA5: {ema5}. EMA8: {ema8}. EMA100: {ema100}[/green]')
                # Wait for the next interval (e.g., 5 minute)
                console.print('[yellow]Waiting for the next candle...[/yellow]')

            elif len(positions_recent) > 0:
                print(f'Price Entered: {entry_prices}, Stop Loss: {sl_price_set}, Side: {side}')
                if side[0]==1 and actions[-2]==-1:
                    if entry_prices[0]<current_price:
                        for position in positions_recent:
                            position_size = position['contracts']
                            tp_order = place_tp_in_buy(symbol_2, current_price, position_size, marginmode)
                            tp_order_id = tp_order['data'][0]['algoId']
                            print('Close Position with Positive PNL is placed.')
                            while True:
                                positions_recent = okx.fetch_positions()
                                if len(positions_recent) == 0:
                                    try:
                                        algo_orders1 = [
                                            {"instId": symbol_2, "algoId": stop_alg_id[0]}
                                        ]
                                        result1 = tradeAPI.cancel_algo_order(algo_orders1)
                                    except Exception as e:
                                        print('Error Skipped')

                                    try:
                                        algo_orders2 = [
                                            {"instId": symbol_2, "algoId": tp_order_id}
                                        ]
                                        result2 = tradeAPI.cancel_algo_order(algo_orders2)
                                    except Exception as e:
                                        print('Error Skipped')

                                    print('Position Closed with Positive PNL.')
                                    entry_prices.clear()
                                    sl_price_set.clear()
                                    tp_price_set.clear()
                                    side.clear()
                                    order_alg_id.clear()
                                    stop_alg_id.clear()
                                    tp_alg_id.clear()
                                    break

                    elif entry_prices[0]>current_price:
                        for position in positions_recent:
                            position_size = position['contracts']
                            sl_order = place_stop_loss_in_buy(symbol_2, current_price, position_size, marginmode)
                            sl_order_id = sl_order['data'][0]['algoId']
                            print('Close Position with Negative PNL is placed.')
                            while True:
                                positions_recent = okx.fetch_positions()
                                if len(positions_recent) == 0:
                                    try:
                                        algo_orders1 = [
                                            {"instId": symbol_2, "algoId": stop_alg_id[0]}
                                        ]
                                        temp_sl_cancel1 = tradeAPI.cancel_algo_order(algo_orders1)
                                    except Exception as e:
                                        print('Error Skipped')

                                    try:
                                        algo_orders2 = [
                                            {"instId": symbol_2, "algoId": sl_order_id}
                                        ]
                                        temp_sl_cancel2 = tradeAPI.cancel_algo_order(algo_orders2)
                                    except Exception as e:
                                        print('Error Skipped')

                                    print('Position Closed with Negative PNL.')
                                    entry_prices.clear()
                                    sl_price_set.clear()
                                    tp_price_set.clear()
                                    side.clear()
                                    order_alg_id.clear()
                                    stop_alg_id.clear()
                                    tp_alg_id.clear()
                                    break

                if side[0] == -1 and actions[-2] == -1:
                    if entry_prices[0] > current_price:
                        for position in positions_recent:
                            position_size = position['contracts']
                            tp_order = place_tp_in_sell(symbol_2, current_price, position_size, marginmode)
                            tp_order_id = tp_order['data'][0]['algoId']
                            print('Close Position with Positive PNL is placed.')
                            while True:
                                positions_recent = okx.fetch_positions()
                                if len(positions_recent) == 0:

                                    try:
                                        algo_orders1 = [
                                            {"instId": symbol_2, "algoId": stop_alg_id[0]}
                                        ]
                                        result = tradeAPI.cancel_algo_order(algo_orders1)
                                    except Exception as e:
                                        print('Error Skipped')

                                    try:
                                        algo_orders2 = [
                                            {"instId": symbol_2, "algoId": tp_order_id}
                                        ]
                                        result = tradeAPI.cancel_algo_order(algo_orders2)
                                    except Exception as e:
                                        print('Error Skipped')

                                    print('Position Closed with Positive PNL.')
                                    entry_prices.clear()
                                    sl_price_set.clear()
                                    tp_price_set.clear()
                                    side.clear()
                                    order_alg_id.clear()
                                    stop_alg_id.clear()
                                    tp_alg_id.clear()
                                    break
                    elif entry_prices[0] < current_price:
                        for position in positions_recent:
                            position_size = position['contracts']
                            sl_order = place_stop_loss_in_sell(symbol_2, current_price, position_size, marginmode)
                            sl_order_id = sl_order['data'][0]['algoId']
                            print('Close Position with Negative PNL is placed.')
                            while True:
                                positions_recent = okx.fetch_positions()
                                if len(positions_recent) == 0:

                                    try:
                                        algo_orders1 = [
                                            {"instId": symbol_2, "algoId": stop_alg_id[0]}
                                        ]
                                        temp_sl_cancel = tradeAPI.cancel_algo_order(algo_orders1)
                                    except Exception as e:
                                        print('Error Skipped')

                                    try:
                                        algo_orders2 = [
                                            {"instId": symbol_2, "algoId": sl_order_id}
                                        ]
                                        temp_sl_cancel = tradeAPI.cancel_algo_order(algo_orders2)
                                    except Exception as e:
                                        print('Error Skipped')

                                    print('Position Closed with Negative PNL.')
                                    entry_prices.clear()
                                    sl_price_set.clear()
                                    tp_price_set.clear()
                                    side.clear()
                                    order_alg_id.clear()
                                    stop_alg_id.clear()
                                    tp_alg_id.clear()
                                    break

        except Exception as e:
            print(e)
        sync_to_5_minute()
        time.sleep(3)


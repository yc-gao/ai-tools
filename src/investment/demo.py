import math

import yfinance as yf
import matplotlib.pyplot as plt

class TradingFeeCalculator:
    def __init__(self, buy_fee_percentage=0, sell_fee_percentage=0):
        self.buy_fee_percentage = buy_fee_percentage
        self.sell_fee_percentage = sell_fee_percentage

    def buy(self, reference_amount, stock_price):
        stock_amount = math.floor(reference_amount / stock_price)
        return reference_amount - stock_amount * stock_price, stock_amount, 0  # No fees for simplicity

    def sell(self, stock_amount, stock_price):
        return stock_amount * stock_price, 0, 0  # No fees for simplicity


class OverNightTradingPolicy:
    def __init__(self, fee_calculator, amount, stock_amount=0, sell_threshold=0.04, buy_threshold=-0.02):
        self.fee_calculator = fee_calculator
        self.amount = amount
        self.stock_amount = stock_amount
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold

        self.fee_amount = 0
        self.prev_day_price = None

    def update_tick(self, price, is_last_tick=False):
        if self.stock_amount == 0:
            return

        is_sell_condition_met = price / self.prev_day_price - 1 > self.sell_threshold
        if is_sell_condition_met or is_last_tick:
            amount, stock_amount, fee = self.fee_calculator.sell(self.stock_amount, price)
            self.amount += amount
            self.stock_amount = stock_amount
            self.fee_amount += fee

    def update_day(self, price, is_last_day=False):
        # If it's the first day, we don't have a previous day's price to compare with, so we just set the previous day's price and return.
        if self.prev_day_price is None:
            self.prev_day_price = price
            return

        if is_last_day:
            self.update_tick(price, is_last_tick=True)
        
        if not is_last_day:
            if (price / self.prev_day_price - 1) < self.buy_threshold:
                # If the price has dropped by more than the buy threshold compared to the previous day's price, we buy all stocks.
                amount, stock_amount, fee = self.fee_calculator.buy(self.amount, price)
                self.amount = amount
                self.stock_amount += stock_amount
                self.fee_amount += fee
        
        self.prev_day_price = price

class TradingBacktest:
    def __init__(self, trading_policy):
        self.trading_policy = trading_policy

        self.amount_history = []
        self.stock_amount_history = []
        self.asset_history = []

    def run(self, data):
        for row in data.itertuples():
            is_last_day = row.Index == data.index[-1]

            self.trading_policy.update_tick(row.Open)
            self.trading_policy.update_tick((row.High + row.Open) / 2)
            self.trading_policy.update_tick(row.High)
            self.trading_policy.update_tick((row.High + row.Close) / 2)
            self.trading_policy.update_tick(row.Close)
            self.trading_policy.update_day(row.Close, is_last_day=is_last_day)

            self.amount_history.append(self.trading_policy.amount)
            self.stock_amount_history.append(self.trading_policy.stock_amount)
            self.asset_history.append(self.trading_policy.amount + self.trading_policy.stock_amount * row.Close)
            
            if is_last_day:
                break

        return self.trading_policy.amount

initial_amount = 100_0000
fee_calculator = TradingFeeCalculator()
trading_policy = OverNightTradingPolicy(fee_calculator, initial_amount, 0,  0.04, -0.01)
backtest = TradingBacktest(trading_policy)

ticker = yf.Ticker("588060.SS")
data = ticker.history(period="1y")
backtest.run(data)

fig, ax1 = plt.subplots()
ax1.plot(data.index, backtest.asset_history, label='Total Asset Value')

ax2 = ax1.twinx()
ax2.plot(data.index, data['Close'], label='Stock Price', color='orange')

plt.show()

# rate = data['High'][1:] / data['Close'][:-1].values
# print(rate.prod())
# rate.hist(bins=100)
# plt.show()
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Project Description:
This project aims to analyze the historical stock data of major banks, 
including Bank of America (BAC), Citigroup (C), Goldman Sachs (GS), JPMorgan Chase (JPM), Morgan Stanley (MS), and Wells Fargo (WFC). 
The analysis covers various aspects such as closing prices, trading volumes, daily returns, risk metrics, moving averages, and cumulative returns.

Steps and Analysis:

1. Data Loading and Preparation:
   - Historical stock data for the selected banks is fetched using the yfinance library for the period from January 1, 2023, to January 1, 2024.
   - The fetched data is concatenated into a single DataFrame, with keys corresponding to the ticker symbols of the banks.

2. Descriptive Statistics:
   - The project begins by displaying the first few rows of the DataFrame and providing basic statistical descriptions of the data.

3. Data Visualization:
   - Closing Prices: Visualizes the closing prices of all the banks over time.
   - Trading Volumes: Visualizes the trading volumes of all the banks over time.

4. Correlation Analysis:
   - Calculates daily returns for each stock and then computes the correlation matrix of these returns.
   - A heatmap is plotted to visualize the correlation between the different bank stocks.

5. Returns Analysis:
   - Daily Returns: Plots the daily returns of each stock to observe the day-to-day performance.
   - Histogram: Plots a histogram of the daily returns for JPMorgan Chase (JPM) to understand the distribution of returns.

6. Risk Analysis:
   - Standard Deviation (Volatility): Calculates and prints the standard deviation of daily returns, which measures the volatility of each stock.
   - Value at Risk (VaR): Calculates and prints the 5% Value at Risk, indicating the potential loss with a probability of 0.05.
   - Sharpe Ratio: Calculates and prints the Sharpe Ratio, which measures the risk-adjusted return of each stock.
   - Rolling 30-Day Volatility: Visualizes the rolling 30-day standard deviation of daily returns to show how volatility changes over time.

7. Cumulative Returns:
   - Calculates and plots the cumulative returns of each stock, showing the total return over time considering the compounding of daily returns.

8. Maximum Drawdown:
   - Calculates and prints the maximum drawdown for each stock, indicating the largest peak-to-trough decline.
   - Plots the drawdown for each stock to visualize periods of significant decline.

9. Moving Averages:
   - Calculates the 30-day and 90-day Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) for each stock.
   - Plots the closing prices along with the calculated SMAs and EMAs to analyze trends and potential support/resistance levels.
"""


# Define the start and end dates
start = datetime.datetime(2023, 1, 1)
end = datetime.datetime(2024, 1, 1)

# List of stock ticker symbols
tickers = ["BAC", "C", "GS", "JPM", "MS", "WFC"]

# Fetch stock data for each ticker
data = {}
for ticker in tickers:
    data[ticker] = yf.download(ticker, start=start, end=end)

# ** Use pd.concat to concatenate the bank dataframes together to a single data frame called bank_stocks.
# Set the keys argument equal to the tickers list. Also pay attention to what axis you concatenate on.**
bank_stocks = pd.concat(data.values(), axis=1, keys=tickers)
bank_stocks.columns.names = ["Bank Ticker", "Stock Info"]
print(bank_stocks.head())
print(bank_stocks.describe())

# Data Visualization for Closing Prices
plt.figure(figsize=(12, 6))
for ticker in tickers:
    bank_stocks[ticker]["Close"].plot(label=ticker)
plt.title("Bank Stocks Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Data Visualization for Trading Volumes
plt.figure(figsize=(12, 6))
for ticker in tickers:
    bank_stocks[ticker]["Volume"].plot(label=ticker)
plt.title("Bank Stocks Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.show()

# Correlation Analysis
returns = bank_stocks.xs("Close", level="Stock Info", axis=1).pct_change()
correlation_matrix = returns.corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Bank Stocks")
plt.show()

# Returns Analysis
daily_returns = returns

plt.figure(figsize=(12, 6))
for ticker in tickers:
    daily_returns[ticker].plot(label=ticker)
plt.title("Daily Returns of Bank Stocks")
plt.xlabel("Date")
plt.ylabel("Daily Returns")
plt.legend()
plt.show()

# JPM Daily Returns Histogram
selected_stock = "JPM"
daily_returns[selected_stock].hist(bins=50, alpha=0.7)
plt.title(f"Daily Returns Histogram for {selected_stock}")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.show()


## Descriptive Statistics on Risk Analysis
# Standard Deviation (Volatility)
std_dev = daily_returns.std()
print("Standard Deviation of Daily Returns:")
print(std_dev)

# Value at Risk (VaR) at the 5% level
VaR = daily_returns.quantile(0.05)
print("Value at Risk (VaR) at 5% level:")
print(VaR)

# Sharpe Ratio: Measures risk-adjusted return.
# provides a way to understand how much return an investment is generating per unit of risk.
# A higher Sharpe Ratio indicates that the investment is providing a better return for the level of risk taken.
risk_free_rate = 0
sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std()
print("Sharpe Ratio:")
print(sharpe_ratio)

# Rolling 30-Day Volatility:
# Shows how volatility changes over time, which helps in understanding the stability of the stock.
rolling_volatility = daily_returns.rolling(window=30).std()
plt.figure(figsize=(12, 6))
for ticker in tickers:
    rolling_volatility[ticker].plot(label=ticker)
plt.title("Rolling 30-Day Volatility of Bank Stocks")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()

# Cumulative Returns:
# Shows the total return of the investment over time
# helping to assess the overall performance.
cumulative_returns = (1 + daily_returns).cumprod() - 1
"""
1 + daily_returns: Adding 1 to the daily returns transforms them into a format suitable for calculating cumulative returns. 
If the daily return for a stock on a given day is r, then 1 + r represents the growth factor for that day 

.cumprod(): This method calculates the cumulative product of the growth factors along the specified axis (by default, axis=0 for rows). 

- 1: Subtracting 1 adjusts the cumulative product back to a return format. 
The cumulative return for a stock is the total return over the period, considering all the individual daily returns compounded together. """


plt.figure(figsize=(12, 6))
for ticker in tickers:
    cumulative_returns[ticker].plot(label=ticker)
plt.title("Cumulative Returns of Bank Stocks")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()

# Maximum Drawdown:
# Indicates the largest peak-to-trough decline, which is useful for understanding potential losses.
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown = drawdown.min()
print("Maximum Drawdown:")
print(max_drawdown)
plt.figure(figsize=(12, 6))
for ticker in tickers:
    drawdown[ticker].plot(label=ticker)
plt.title("Drawdown of Bank Stocks")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.show()

# MA: Moving Averages
# Windows
ma_window_30 = 30
ma_window_90 = 90

# SMA and EMA
# EMA: putting more weight to recent prices.
for ticker in tickers:
    bank_stocks[ticker, "SMA30"] = (
        bank_stocks[ticker]["Close"].rolling(window=ma_window_30).mean()
    )
    bank_stocks[ticker, "SMA90"] = (
        bank_stocks[ticker]["Close"].rolling(window=ma_window_90).mean()
    )
    bank_stocks[ticker, "EMA30"] = (
        bank_stocks[ticker]["Close"].ewm(span=ma_window_30, adjust=False).mean()
    )  # ewm stands for Exponential Weighted Moving Average
    bank_stocks[ticker, "EMA90"] = (
        bank_stocks[ticker]["Close"].ewm(span=ma_window_90, adjust=False).mean()
    )

# Plot Moving Averages
for ticker in tickers:
    plt.figure(figsize=(12, 6))
    bank_stocks[ticker]["Close"].plot(label="Close Price")
    bank_stocks[ticker, "SMA30"].plot(label="30-Day SMA")
    bank_stocks[ticker, "SMA90"].plot(label="90-Day SMA")
    bank_stocks[ticker, "EMA30"].plot(label="30-Day EMA")
    bank_stocks[ticker, "EMA90"].plot(label="90-Day EMA")
    plt.title(f"Moving Averages of {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# stock_bot

## What is this bot?

This python script uses an LSTM neural net to analyze n day long windows of closing stock prices and produce a next day prediction of the closing price of selected stock(s). The stock data is downloaded using yfinance and the ticker names given in the config file. From this the bot produces a stock buy/sell decision based on if our prediction is higher or lower than the current stock price. Finally the bot executes the decision as an Alpaca paper trade. The bot runs the morning after every market day.

## Setup

In order to use this bot one must make an Alpaca paper trading account and provide api keys in the config file. There is a default config file in src, copy it to your home directory and remove 'default_' in the file name for the script to recognize it.

## Sysargs

#1 --config= Will take a custom config file path for the script to use

#2 --test_mode Will immediately start analysis and trade where normally the bot waits for market close

#3 --enable_trading Will allow the bot to place paper trades after analysis

# stock_bot
This python script uses an LSTM neural net to produce stock buying decisions and executes them as Alpaca paper trades. This bot runs the morning after every market day.
In order to use this bot one must make an Alpaca paper trading account and provide api keys in the config file. There is a default config file in src, copy it to your home directory and remove 'default_' in the file name for the script to recognize it.
--config= Will take a config file path for the script to use
--test_mode Will immediately start analysis and trade where normally the bot waits for market close
--enable_trading Will allow the bot to place paper trades after analysis

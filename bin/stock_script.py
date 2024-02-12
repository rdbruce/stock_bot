#!/usr/bin/env python3

import argparse
import os
import sys
from pprint import pprint
from pathlib import Path
from datetime import datetime
from time import sleep
import logging

logging.basicConfig(filename=f'{os.environ["HOME"]}/stock_log.log', level=logging.INFO)

def analyze_and_trade():
    processor = TradingApi(config)
    predictor = StockPredictor(config)
    pprint(predictor.prediction_container())

    trades = predictor.get_trades()
    #pprint(trades)

    if args.enable_trading is True:
        processor.post_next_day_trades(trades)

#Enter args
parser = argparse.ArgumentParser(description='Process stock data')
parser.add_argument('--config', default=f'{os.environ["HOME"]}/stock_config.yaml')
parser.add_argument('--enable_trading', action='store_true')
parser.add_argument('--test_mode', action='store_true')
args = parser.parse_args()

#Set PATH to import module
package_dir = Path(sys.argv[0]).resolve().parent.parent
sys.path.append(package_dir.as_posix())
from src import *

#Get config
try:
  config = get_stock_config(args.config)
except:
  print('No config file found. Find a default in /src and copy to home dir.')
  quit()

if args.test_mode is True:
    analyze_and_trade()
    quit()
else:
    while True:
        # delay until next time day equals 6am
        now = datetime.now()
        next_run = now.replace(day=now.day+1, hour=1, minute=0, second=0, microsecond=0)
        # next_run = now.replace(minute=now.minute+1, second=0, microsecond=0)
        while next_run.weekday() in [ 5, 6 ]:
            next_run = next_run.replace(day=next_run.day+1)
        time_diff = (next_run - now).total_seconds()
        print(f'waiting for {time_diff}')
        sleep(time_diff)
        print(datetime.now())
        analyze_and_trade()
    quit()


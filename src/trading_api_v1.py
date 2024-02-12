from .stock_predictor_v1 import TradeType
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import logging

class TradingApi:
    def __init__(self, cfgFile):
        try:
            self.trading_client = TradingClient(cfgFile.api_key, cfgFile.secret_api_key)
        except:
            print('Alpaca client failed to initialize. Check api keys in config file.')
            quit()

    def post_next_day_trades(self, trades):
        for trade in trades:
            print(f'processing {trade}')
            if trade.trade_type == TradeType.BUY:
                market_order_data = MarketOrderRequest(
                    symbol=trade.ticker,
                    qty=trade.num_shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY)
                # Place market order
                print(market_order_data)
                logging.info(market_order_data)
                market_order = self.trading_client.submit_order(order_data=market_order_data)
            elif trade.trade_type == TradeType.SELL:
                market_order_data = MarketOrderRequest(
                    symbol=trade.ticker,
                    qty=trade.num_shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY)
                # Place market order
                print(market_order_data)
                logging.info(market_order_data)
                market_order = self.trading_client.submit_order(order_data=market_order_data)
            else:
                print('No trades for today.')
                logging.info('No trades for today.')
            

        portfolio = self.trading_client.get_all_positions()
        # Print the quantity of shares for each position.
        for position in portfolio:
            print("{} shares of {}".format(position.qty, position.symbol))
            logging.info("{} shares of {}".format(position.qty, position.symbol))



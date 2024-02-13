from enum import Enum
from dataclasses import dataclass

import time as tm
import datetime as dt
from pprint import pprint
import matplotlib.pyplot as plt
import tensorflow as tf

# Data aquisition and preparation
import yaml
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# AI
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


class TradeType(Enum):
    SIT = 0
    BUY = 1
    SELL = 2

@dataclass
class Trade:
    ticker: str
    trade_type: TradeType
    num_shares: float

class StockPredictor:
    def __init__(self, cfg):
        # Save config parameters
        self.cfg = cfg
        #Define model
        self.model = Sequential()
        self.model.add(LSTM(60, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(120, return_sequences=False))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(20))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def grab_data(self):
        date_today = tm.strftime('%Y-%m-%d')
        date_of_start = (dt.date.today() - dt.timedelta(days=self.cfg.num_days_concerned)).strftime('%Y-%m-%d')
        self.stock_data = yf.download(self.cfg.ticker_list, start=date_of_start, end=date_today, interval = "1d")

        self.stock_data = self.stock_data.drop(['Close', 'High', 'Low', 'Open', 'Volume'], axis = 1)
        self.stock_data.columns = self.stock_data.columns.droplevel()

        #Initialize our x axis(for plotting)
        self.x_time = self.stock_data.index[self.stock_data.shape[0] - self.cfg.test_days : self.stock_data.shape[0]]

    def mk_tnsr(self, stock_to_predict):#Create target_column
        target_column  = self.stock_data[stock_to_predict]
        target_column = pd.DataFrame({stock_to_predict: target_column })
        target_column = target_column.iloc[7:]
        self.today_price = np.squeeze(target_column.tail(1).to_numpy())

        #Scale the main body and target stock
        self.x_scaler = MinMaxScaler()
        main_body = pd.DataFrame(self.x_scaler.fit_transform(self.stock_data))
        self.y_scaler = MinMaxScaler()
        target_column = self.y_scaler.fit_transform(target_column)

        #Make Our Tensor
        num_windows = main_body.shape[0] - self.cfg.window_size + 1
        tensor = np.zeros((num_windows, self.cfg.window_size, main_body.shape[1]))

        # Populate the tensor with windows
        for i in range(num_windows):
            window = main_body.iloc[i:i+self.cfg.window_size, :].to_numpy()
            tensor[i, :, :] = window

        # Seperate into train and test
        total_rows = tensor.shape[0]

        self.x_train = tensor[: total_rows - self.cfg.test_days, :, :]
        self.x_test = tensor[total_rows - self.cfg.test_days : total_rows, :, :]
        self.y_train = target_column[: total_rows - self.cfg.test_days, :]
        self.y_test = np.ndarray((0,0))
        if self.cfg.test_days != 1:
            y_test = target_column[total_rows - self.cfg.test_days : total_rows, :]
            self.y_test = self.y_scaler.inverse_transform(y_test)

    def get_trained_model(self):
        self.model.fit(self.x_train, self.y_train,
                batch_size=self.cfg.batch_size,
                epochs=self.cfg.epochs,
                verbose=1)
        self.model.summary()
    
    def get_predictions(self):
        y_predicted = self.model.predict(self.x_test)
        pprint(y_predicted.shape)
        self.y_predicted_transformed = self.y_scaler.inverse_transform(y_predicted)

        self.tomorrow_price = np.squeeze(self.y_predicted_transformed[-1])
    
    def prediction_container(self):
        outgoing_trades = []
        self.grab_data()
        self.cfg.test_days = 1

        for stock_to_predict in self.cfg.stocks_to_predict:
            self.mk_tnsr(stock_to_predict)
            self.get_trained_model()
            self.get_predictions()
            outgoing_trades.append(self.form_trade(stock_to_predict))

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Outgoing trades:')
        pprint(outgoing_trades)
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        return outgoing_trades
    
    def test_container(self):
        self.grab_data()

        for stock_to_predict in self.cfg.stocks_to_predict:
            self.mk_tnsr(stock_to_predict)
            print('-----------------------------------------------------------------------------------------------------')
            pprint(self.x_train.shape)
            pprint(self.y_train.shape)
            print('-----------------------------------------------------------------------------------------------------')
            pprint(self.x_test.shape)
            pprint(self.y_test.shape)
            print('-----------------------------------------------------------------------------------------------------')
            self.get_trained_model()
            self.get_predictions()
            print("RMSE for neural net:", np.sqrt(np.mean((self.y_test - self.y_predicted_transformed[:-1])**2)))
    
    def form_trade(self, stock_to_predict):
        outgoing_trade = Trade(stock_to_predict, TradeType.SIT, 1)

        print(f'Todays Price {self.today_price}')
        print(f'Tomorrows Predicted Price {self.tomorrow_price}')
        if self.today_price + 2 < self.tomorrow_price:
            outgoing_trade.trade_type = TradeType.BUY
        elif self.today_price > self.tomorrow_price + 2:
            outgoing_trade.trade_type = TradeType.SELL
        else:
            pass

        outgoing_trade.num_shares = self.cfg.trading_quanta/self.today_price

        return outgoing_trade
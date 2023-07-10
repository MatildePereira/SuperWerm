import datetime

import yfinance as yf
import numpy as np
from keras.layers import BatchNormalization, LSTM, Dense, Dropout, Input, Concatenate, Flatten
from keras.models import Model
from keras.optimizers import Adam
import tensorflow


class Trader:
    def __init__(self, model, init_balance=100, stock_history=[], companies=['GOOG', 'AAPL'],
                 interval="1h", buy_tax=0.06, timesteps=10, batch_size=6, pessimism_factor=0.03, hold_reward=0.3,hold_hourly_decay=0.03, learning_rate=0.001):
        self.init_balance = init_balance
        self.balance = init_balance
        self.model = model
        self.stock_history = stock_history
        self.company_amount = len(companies)
        self.wallet = {}
        for i in companies:
            self.wallet[i] = [-1, 0]
        self.buy_screenshots = {}
        self.interval = interval
        self.buy_tax = buy_tax
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.history = []
        self.pessimism_factor = pessimism_factor
        self.sell_prices = {}
        self.hold_reward = hold_reward
        self.hold_hourly_decay = hold_hourly_decay
        self.learning_rate = learning_rate
        self.train_x = []
        self.train_y = []

    def buy(self, company, amount):
        """
        :param company: indicator of which company you're buying from
        :param amount: quantity of stocks to buy
        :void
        """
        stock_data = self.get_stock_data(company)
        buy_price = (stock_data['High'].values[0] + stock_data['Low'].values[0])*(1+self.pessimism_factor)*(1+self.buy_tax)/2
        self.wallet[company][1] += amount
        self.wallet[company][0] = buy_price
        self.balance -= buy_price*amount

    def sell(self, company, amount):
        """
        :param company: indicator of which company you're selling the stock
        :param amount: quantity of stocks to sell
        :void
        """
        stock_data = self.get_stock_data(company)
        sell_price = (stock_data['High'].values[0] + stock_data['Low'].values[0])/(2*(1+self.pessimism_factor))
        self.sell_prices[company] = sell_price
        self.wallet[company][1] -= amount
        self.balance += sell_price*amount

    def get_stock_data(self, company, period="immediate"):
        comp = yf.Ticker(company)
        if (period != "immediate"):
            hist = comp.history(period=period, interval=self.interval)
        else:
            hist = comp.history(period="1d", interval="1m").values[-1]
        return hist

    def stock_data_prepare(self, period="1mo"):
        data = {}  # Cada dado de cada empresa tem tamanho [timesteps, features]
        for company in self.wallet.keys():
            hist = self.get_stock_data(company, period=period)[['Open', 'High', 'Low', 'Close', 'Volume']].values
            xin = []
            for i in range(self.timesteps, len(hist)):
                xin.append(hist[i - self.timesteps:i])
            xin = np.array(xin)
            xin = xin.reshape(xin.shape[0], xin.shape[1], 5)  # este 5 é os 5 valores da stock
            data[company] = xin
        return data

    def decide_transaction(self):
        input_data = self.stock_data_prepare().values + [self.wallet.values]
        results = self.model.predict(input_data)
        '''
        0 - B - BUY
        1 - S - SELL
        2 - H - HOLD
        '''
        train_queue = []
        for i in range(len(results)):
            decision = np.argmax(results[i])
            if decision == 0 and self.wallet.values[i][1] == 0:
                self.buy(self.wallet.keys()[i])
                self.buy_screenshots[self.wallet.keys()[i]] = [input_data, results[i], datetime.datetime.now()]
            elif decision == 1 and self.wallet.values[i][1] > 0:
                self.sell(self.wallet.keys()[i])
                train_queue.append("S")
            else:
                train_queue.append("H")
        if len(train_queue) > 0:
            self.add_to_training_data(results, train_queue, input_data)

    def add_to_training_data(self, results, train_queue, input_data):
        q_values_updated = results
        for i in range(len(train_queue)):
            if train_queue[i] == "H":
                q_values_updated[i][2] += self.learning_rate*(self.hold_reward*(1-self.hold_hourly_decay**((datetime.datetime.now() - self.buy_screenshots[self.wallet.keys()[i]][2]).total_seconds()/60)))
            elif train_queue[i] == "S":
                q_values_updated[i][1] += self.learning_rate*((self.sell_prices[self.wallet.keys()[i]] - self.wallet[self.wallet.keys()[i]][0])/self.wallet[self.wallet.keys()[i]][0])
        self.train_x.append(input_data)
        self.train_y.append(q_values_updated)

    def create_model(self, stock_correlation_sizes=[300, 200, 100], wallet_correlation_sizes=[50, 30, 10],
                     prediction_sizes=[200, 100, 100], decision_sizes=[200, 100]):
        inputs = []
        stock_inputs=[]
        input2 = Input(shape=(len(self.wallet.keys()), 2,), name="Wallet Input")
        wallet_input = BatchNormalization(synchronized=True)(input2)

        for i in range(len(self.wallet.keys())):
            inputs.append(Input(shape=(self.timesteps, 5,), name="Stock input "+str(i)))
            stock_inputs.append(BatchNormalization(synchronized=True)(inputs[i]))
        big_boy = Concatenate()(stock_inputs)
        for size in stock_correlation_sizes[:-1]:
            big_boy = LSTM(units=size, return_sequences=True)(
                big_boy)  # big_boy = LSTM(units=size, input_shape=(self.timesteps, 5))(big_boy)
        big_boy = LSTM(units=stock_correlation_sizes[-1])(big_boy)

        wallet_boy = Dense(units=wallet_correlation_sizes[0])(wallet_input)
        for size in wallet_correlation_sizes[1:]:
            wallet_boy = Dense(units=size, activation="linear")(wallet_boy)

        outputs = []
        if len(prediction_sizes) > 0:
            prediction_boys = [None for i in range(len(stock_inputs))]
            for i in range(len(stock_inputs)):
                prediction_boys[i] = LSTM(units=prediction_sizes[0], return_sequences=True)(stock_inputs[
                                                                                                i])  # prediction_boys[i] = LSTM(units=prediction_sizes[0], input_shape=(self.timesteps, 5))(inputs[i])

                for size in prediction_sizes[1:-1]:
                    prediction_boys[i] = LSTM(units=size, return_sequences=True)(prediction_boys[i])
                prediction_boys[i] = LSTM(units=size)(prediction_boys[i])

            # intermediate_layers = [Flatten()(prediction_boy) for prediction_boy in prediction_boys] + [ Flatten()(
            # wallet_boy)] + [ Flatten()(big_boy)]
            intermediate_layers = []
            for prediction_boy in prediction_boys:
                intermediate_layers.append([Flatten()(wallet_boy)] + [Flatten()(big_boy)] + [Flatten()(prediction_boy)])
            for layer in intermediate_layers:
                decision_boy = Concatenate(axis=1)(layer)
                for size in decision_sizes:
                    decision_boy = Dense(size, activation="linear")(decision_boy)
                decision_boy = Dense(3)(decision_boy)
                outputs.append(decision_boy)
        else:
            intermediate_layers = [Flatten()(wallet_boy)] + [Flatten()(big_boy)]
            decision_boy = Concatenate(axis=1)(intermediate_layers)
            for size in decision_sizes:
                decision_boy = Dense(size, activation="linear")(decision_boy)
            decision_boy = Dense(3)(decision_boy)
            outputs.append(decision_boy)

        self.model = Model(inputs=inputs + [input2], outputs=outputs)
        opt = Adam(learning_rate=self.learning_rate)

        self.model.compile(optimizer=opt, loss="mse")
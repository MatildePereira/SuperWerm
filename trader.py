import yfinance as yf
import numpy as np
from keras.layers import BatchNormalization, LSTM, Dense, Dropout, Input, Concatenate, Flatten
from keras.models import Model
import tensorflow
class Trader:
    def __init__(self, model, init_balance=100, stock_history=[], companies=['GOOG', 'AAPL'],
                 interval="1h", buy_tax=0.06, timesteps=30, batch_size =6):
        self.init_balance = init_balance
        self.balance = init_balance
        self.model = model
        self.stock_history = stock_history
        self.company_amount = len(companies)
        self.wallet = {}
        for i in companies:
            self.wallet[i] = [None, 0]
        self.interval = interval
        self.buy_tax = buy_tax
        self.timesteps=timesteps
        self.batch_size = batch_size
    def buy(self, company, amount):
        """
        :param company: indicator of which company you're buying from
        :param amount: quantity of stocks to buy
        :void
        """
        self.wallet[company][1] += amount

    def sell(self, company, amount):
        """
        :param company: indicator of which company you're selling the stock
        :param amount: quantity of stocks to sell
        :void
        """
        self.wallet[company][1] -= amount

    def get_stock_data(self, company, period="immediate"):
        comp = yf.Ticker(company)
        if (period != "immediate"):
            hist = comp.history(period=period, interval=self.interval)
        else:
            hist = comp.history(period="1d", interval="1m").values[-1]
        return hist
    def stock_data_prepare(self, period="1mo"):
        data = {} #Cada dado de cada empresa tem tamanho [timesteps, features]
        for company in self.wallet.keys():
            hist = self.get_stock_data(company, period=period)[['Open', 'High', 'Low','Close','Volume']].values
            xin = []
            for i in range(self.timesteps, len(hist)):
                xin.append(hist[i-self.timesteps:i])
            xin = np.array(xin)
            xin = xin.reshape(xin.shape[0], xin.shape[1], 5) #este 5 é os 5 valores da stock
            data[company] = xin
        return data
    def create_model(self, stock_correlation_sizes=[300,200,100],wallet_correlation_sizes=[50,30,10],prediction_sizes=[200,100,100],decision_sizes=[200,100]):
        inputs = []
        wallet_input = Input(shape=(len(self.wallet.keys()), 2,))

        wallet_input = BatchNormalization(synchronized=True)(wallet_input)
        for i in range(len(self.wallet.keys())):
            inputs.append(Input(shape=(self.timesteps,5,)))

        big_boy = Concatenate()(inputs)
        big_boy = BatchNormalization(synchronized=True)(big_boy)
        for size in stock_correlation_sizes[:-1]:
            big_boy = LSTM(units=size, return_sequences=True )(big_boy) #big_boy = LSTM(units=size, input_shape=(self.timesteps, 5))(big_boy)
        big_boy = LSTM(units=stock_correlation_sizes[-1])(big_boy)

        wallet_boy = Dense(units=wallet_correlation_sizes[0])(wallet_input)
        for size in wallet_correlation_sizes[1:]:
            wallet_boy = Dense(units=size, activation="linear")(wallet_boy)

        if len(prediction_sizes) > 0:
            prediction_boys = [None for i in range(len(inputs))]
            for i in range(len(inputs)):
                prediction_boys[i] = LSTM(units=prediction_sizes[0], return_sequences=True)(inputs[i])#prediction_boys[i] = LSTM(units=prediction_sizes[0], input_shape=(self.timesteps, 5))(inputs[i])

                for size in prediction_sizes[1:-1]:
                    prediction_boys[i] = LSTM(units=size, return_sequences=True)(prediction_boys[i])
                prediction_boys[i] = LSTM(units=size)(prediction_boys[i])

            intermediate_layers = [Flatten()(prediction_boy) for prediction_boy in prediction_boys] + [ Flatten()(wallet_boy)] + [ Flatten()(big_boy)]
        else:
            intermediate_layers =[Flatten()(wallet_boy)] + [ Flatten()(big_boy)]
        decision_boy = Concatenate(axis=1)(intermediate_layers)
        for size in decision_sizes:
            decision_boy = Dense(size, activation="linear")(decision_boy)
        decision_boy = Dense(3)(decision_boy)

        self.model = Model(inputs = inputs+[wallet_input], outputs=decision_boy)
        #Falta compile






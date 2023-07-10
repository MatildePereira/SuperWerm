import datetime

import yfinance as yf
import numpy as np
from keras.layers import BatchNormalization, LSTM, Dense, Dropout, Input, Concatenate, Flatten
from keras.models import Model
from keras.optimizers import Adam
import tensorflow


class Trader:
    def __init__(self, model, init_balance=100, stock_history=[], companies=['GOOG', 'AAPL'],
                 interval="1h", buy_tax=0.06, timesteps=10, batch_size=6, pessimism_factor=0.03, hold_reward=0.3,hold_periodic_decay=0.03, learning_rate=0.001, forbidden_sell_punishment=1):
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
        self.hold_reward = hold_reward
        self.hold_periodic_decay = hold_periodic_decay
        self.learning_rate = learning_rate
        self.history = [] #[input_data, [output_data], [transactions], [rewards]]
        self.forbidden_sell_punishment = forbidden_sell_punishment
    def get_buy_price(self,stock_data):
        return (stock_data['High'].values[0] + stock_data['Low'].values[0])*(1+self.pessimism_factor)*(1+self.buy_tax)/2
    def get_sell_price(self, stock_data, isList=False):
        if not isList:
            return (stock_data['High'].values[0] + stock_data['Low'].values[0])/(2*(1+self.pessimism_factor))
        else:
            return (stock_data[0][-1][1] + stock_data[0][-1][1][2])/(1*(1+self.pessimism_factor))
    def buy(self, company, amount):
        """
        :param company: indicator of which company you're buying from
        :param amount: quantity of stocks to buy
        :void
        """
        stock_data = self.get_stock_data(company)
        buy_price = self.get_buy_price(stock_data)
        self.wallet[company][0] = (self.wallet[company][0]*self.wallet[company][1] + buy_price*amount)/(amount + self.wallet[company][1])
        self.wallet[company][1] += amount
        self.balance -= buy_price*amount

    def sell(self, company, amount):
        """
        :param company: indicator of which company you're selling the stock
        :param amount: quantity of stocks to sell
        :void
        """
        stock_data = self.get_stock_data(company)
        sell_price = self.get_sell_price(stock_data)
        self.wallet[company][1] -= amount
        self.balance += sell_price*amount

    def get_stock_data(self, period="immediate"):
        hist = {}
        for company in self.wallet.keys():
            comp = yf.Ticker(company)
            if (period != "immediate"):
                hist[company] = comp.history(period=period, interval=self.interval)
            else:
                hist[company] = comp.history(period="1w", interval="1m").tail(1)
        return hist

    def prepare_stock_data(self, received_data):
        data = {}  # Cada dado de cada empresa tem tamanho [timesteps, features]
        for company in received_data.keys():
            hist = received_data[company][['Open', 'High', 'Low', 'Close', 'Volume']].values
            xin = hist[-(self.timesteps):]
            xin = np.array(xin)
            xin = xin.reshape((1, self.timesteps, 5))  #thang
            data[company] = xin
        return data

    def decide_transaction(self):
        data = self.get_stock_data("1mo")
        input_data = list(self.prepare_stock_data(data).values()) + [list(self.wallet.values())]
        results = self.model.predict(input_data)
        '''
        0 - B - BUY
        1 - S - SELL
        2 - H - HOLD
        '''
        self.history.append([data.index[-1], input_data, results, [("N", 0) for i in len(self.wallet)], np.zeros((len(self.wallet.keys()), 3))])
        '''
        0 - Data
        1 - Input
        2 - Output sem rewards aplicados
        3 - Par letras da transação (B, S, H, S*H) e quantidade para cada empresa
        4 - rewards de cada Q value, a aplicar no fim
        '''
        for i in range(len(results)):
            decision = np.argmax(results[i])
            if decision == 0:
                self.buy(list(self.wallet.keys())[i], 1) #AMOUNT AGORA É 1
                self.history[-1][3][i] = ("B", 1)
            elif decision == 1 and list(self.wallet.values())[i][1] > 0: #aqui podera ser quando vende menos do que tem
                self.sell(list(self.wallet.keys())[i], self.wallet[list(self.wallet.keys())[i]][1]) # AMOUNT AGORA É TUDO
                self.history[-1][3][i] = ("S", self.wallet[list(self.wallet.keys())[i]][1])
                self.update_rewards()
            elif decision == 1 and list(self.wallet.values())[i][1] == 0: #aqui podera ser quando vende mais do que tem
                self.history[-1][3][i] = ("S*H", 0)
            else:
                self.history[-1][3][i] = ("H", 0)

    def update_rewards(self):
        for i in range(len(self.wallet)):
            sell_amount = self.history[-1][3][i][1]
            sell_price = self.get_sell_price(self.history[-1][1][0][i], False)
            self.history[-1][4][i][1] += (self.history[-1][3][i][1]/self.history[-1][1][1][i][1])*(sell_price - self.history[-1][1][1][i][0])/sell_price
            #As que vendeu sobre
            for j in range(2,len(self.history)-1): #self.history[-1] é a ultima venda que desencadeou tudo isto
                if self.history[-j][3][i][0] == "B" and sell_amount > 0:
                    self.history[-j][4][i][0] += (min(self.history[-j][3][i][1], sell_amount)/self.history[-j][1][1][i][1])*(sell_price - self.history[-j][1][1][i][0])/sell_price
                    #Lucro relativo vezes quantas açoes comprou sobre todas das que vendeu
                    sell_amount -= self.history[-j][1][1][i][1]
                elif self.history[-j][3][i][0] == "H" and sell_amount > 0:
                    self.history[-j][4][i][2] = self.hold_reward/((1+self.hold_periodic_decay)**(j-1))
                elif self.history[-j][3][i][0] == "S*H" and sell_amount > 0:
                    self.history[-j][4][i][2] = self.hold_reward/((1+self.hold_periodic_decay)**(j-1))
                    self.history[-j][4][i][1] -= self.forbidden_sell_punishment
                elif self.history[-j][3][i][0] == "S" and sell_amount > 0:
                    continue
                elif self.history[-j][3][i][0] == "S" and sell_amount <= 0:
                    break
        #APENAS PODEMOS TREINAR O MODELO QUANDO TODOS OS INTERVALOS TIVEREM SELL_AMOUNT = SOMATORIO DE STOCKS VENDIDAS PARA TRÁS
        #OU SEJA QUANDO TODAS AS EMPRESAS TIVEREM TODAS AS STOCKS (history[-1][1][i][1]) A 0 AO MESMO TEMPO

    #CRIAR FUNÇAO QUE PEGA NO HISTORICO COMPLETO COM A CONDIÇAO AQUI EM CIMA E FAZ OS UPDATED_Q_VALUES todo
    def create_model(self, stock_correlation_sizes=[300, 200, 100], wallet_correlation_sizes=[50, 30, 10],
                     prediction_sizes=[200, 100, 100], decision_sizes=[200, 100]):
        inputs = []
        stock_inputs=[]
        input2 = Input(shape=(len(self.wallet.keys()), 2,), name="Wallet Input")
        wallet_input = BatchNormalization(synchronized=True)(input2)

        for i in range(len(self.wallet)):
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
import datetime
import time

import tensorflow
import keras
import pandas as pd
import yfinance as yf
import numpy as np
from keras.layers import BatchNormalization, LSTM, Dense, Dropout, Input, Concatenate, Flatten
from keras.models import Model
from keras.optimizers import Adam
from dateutil.relativedelta import relativedelta
import pytz
import copy

'''             ,\
             \\\,_
              \` ,\
         __,.-" =__)
       ."        )
    ,_/   ,    \/\_
    \_|    )_-\ \_-`
jgs    `-----` `--`
'''


class Trader:
    def __init__(self, model_name="JAMEMB", init_balance=100, companies=['GOOG', 'AAPL'],
                 interval="1h", buy_tax=0.06, investible_fraction=0.8, timesteps=10, batch_size=20, pessimism_factor=0,
                 hold_reward=0.05,
                 learning_rate=0.00001, q_learning_rate=0.1, discount_factor=0.1, validation_ratio=0.8, real_time=False,
                 random_choice_chance=0, verbose=0):
        self.validation_ratio = validation_ratio
        self.init_balance = init_balance
        self.balance = init_balance
        self.model_name = model_name
        self.verbose = verbose
        self.wallet = {}
        for i in companies:
            self.wallet[i] = [-1, 0]  # [valor medio investido em compras, quantidade de ações]
        self.interval = interval
        self.buy_tax = buy_tax
        self.investible_fraction = investible_fraction
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.pessimism_factor = pessimism_factor
        self.hold_reward = hold_reward
        self.learning_rate = learning_rate
        self.q_learning_rate = q_learning_rate
        self.discount_factor = discount_factor
        self.history = {}
        self.real_time = real_time
        # self.now = datetime.datetime.strptime('2023-5-12 10:30:00', '%Y-%m-%d %H:%M:%S')
        self.now = pd.Timestamp('2022-01-01 09:30:00-0400', tz='America/New_York')
        # if not self.real_time:
        #   self.now = self.get_stock_data()['GOOG'].index[0] - relativedelta(days=60)  # TODO
        self.random_choice_chance = random_choice_chance
        try:
            self.model = keras.models.load_model(model_name)
            print("FETCHED MODEL: " + model_name)
        except:
            print("CANNOT FECH MODEL " + model_name, ", CREATING NEW...")
            self.create_model()

    def get_buy_price(self, stock_data, is_input_data=False):
        if not is_input_data:
            return (stock_data['High'].values[0] + stock_data['Low'].values[0]) * (1 + self.pessimism_factor) * (
                    1 + self.buy_tax) / 2
        else:
            return (stock_data[-1][-1][1] + stock_data[-1][-1][2]) * (1 + self.pessimism_factor) * (
                    1 + self.buy_tax) / 2

    def get_sell_price(self, stock_data, is_input_data=False):
        if not is_input_data:
            return (stock_data['High'].values[0] + stock_data['Low'].values[0]) / (2 * (1 + self.pessimism_factor))
        else:
            return (stock_data[-1][-1][1] + stock_data[-1][-1][2]) / (2 * (1 + self.pessimism_factor))

    def buy(self, company, amount):
        """
        :param company: indicator of which company you're buying from
        :param amount: quantity of stocks to buy
        :void
        """
        stock_data = self.get_stock_data()[company]
        buy_price = self.get_buy_price(stock_data)
        self.wallet[company][0] = (self.wallet[company][0] * self.wallet[company][1] + buy_price * amount) / (
                amount + self.wallet[company][1])  # Novo preço médio pelo que trocou
        self.wallet[company][1] += amount
        self.balance -= buy_price * amount

        if self.verbose >= 2:
            print("B "+ str(company)+" "+str(buy_price)+ "*"+ str(amount))

    def sell(self, company, amount):
        """
        :param company: indicator of which company you're selling the stock
        :param amount: quantity of stocks to sell
        :void
        """
        stock_data = self.get_stock_data()[company]
        sell_price = self.get_sell_price(stock_data)
        self.wallet[company][1] -= amount
        self.balance += sell_price * amount

        if self.verbose >= 2:
            print("S "+ str(company)+" "+str(sell_price)+ "*"+ str(amount))


    # def get_stock_data(self, immediately=True, points=1, max_margin=2, real_time=False):
    def get_stock_data(self, points=1, max_margin=2):
        """
        Saca os dados de ‘stocks’ das empresas dele
        ALGUNS AVISOS SOBRE O YAHOO FINANCE:
            1. Prazo de 30 dias sob atual para dados de 1 m
            2. Só 7 dias de dados de 1 m é permitido
        :param points: Número de pontos de dados desejados (equivalente a linhas de ações)
        :param max_margin: Margem em meses de onde tirar esses pontos
        :return: dicionário para cada empresa com dataframes
        """

        hist = {}
        end_time = (self.now + relativedelta(days=6)).strftime('%Y-%m-%d')

        # todo:: tf.Ticker() aceita como argumento uma lista de tickers, pode ser mais rapido que o loop mas nao me
        #  apetece ver desta merda

        for company in self.wallet.keys():
            comp = yf.Ticker(company)
            # if immediately:
            if self.real_time:
                start_time = (self.now - relativedelta(days=6)).strftime('%Y-%m-%d')

                # hist[company] = comp.history(start=start_time, interval="1m", end=end_time).tail(1)
                hist[company] = comp.history(start=start_time, interval="1m").tail(points)

            if (not self.real_time) or (points > 1):
                start_time = (self.now - relativedelta(months=max_margin)).strftime('%Y-%m-%d')
                bloop = comp.history(start=start_time, interval=self.interval, end=end_time)
                hist[company] = bloop.loc[(bloop.index <= self.now)].tail(points)

        #fixme: se uma ou duas etc das açoes nao tiverem dados mas as outras sim isto nao dispara mas da erro na mesma
        if np.size(hist.values()) == 0:
            time.sleep(3)
            print("No stock data found, trying again")
            return self.get_stock_data(points, max_margin)
        else:
            return hist

    def prepare_stock_data(self, received_data, size=1):
        """
        Prepara dados
        :param received_data: Dicionário com DataFrames com dados de stocks pra cada empresa
        :param size: Número de dados que queres nesse tight ass
        :return: Dados pro modelo de forma [[size, timesteps, 5]*tamanho_empresas, [size, tamanho_empresas, 2]]
        """
        data = {}  # Cada dado de cada empresa tem tamanho [timesteps, features]

        # [tamanho, 2]
        # [size, tamanho, 2]
        # [3, 2]
        # [1,3,2]
        for company in received_data.keys():
            hist = received_data[company][['Open', 'High', 'Low', 'Close', 'Volume']].values
            data[company] = []
            for i in range(size):
                xin = hist[-self.timesteps:]
                xin = np.array(xin)
                xin = xin.reshape((1, self.timesteps, 5))  # thang
                data[company].append(xin)
                hist = hist[:-1]

        output_which_is_input = []
        for i in data.values():
            output_which_is_input.append(np.concatenate(i))
        # wallet_history = [wallet for wallet in self.history.values()[-size:-1]] + [list(self.wallet.values())]
        wallet_history = [[input_data[0][1] for input_data in
                           list(self.history.values())[len(self.history.values()) - size + 1:]]] + [[
            list(self.wallet.values())]]
        wallet_history = list(filter(None, wallet_history))
        output_which_is_input.append(np.concatenate(wallet_history))
        return output_which_is_input

    def decide_transaction(self):
        """
        Aqui ele decide o que fazer, automaticamente ir buscando os dados mais recentes
        :return: teu cu
        """
        data = self.get_stock_data(points=self.timesteps)
        input = self.prepare_stock_data(data)
        results = self.model.predict(input, verbose=0)
        '''
        Results hopefully terá as dimensões (índice de empresa, índice de transação)
        '''

        '''
        0 - B - BUY
        1 - S - SELL
        2 - H - HOLD
        '''
        t = data[list(self.wallet.keys())[-1]].index[-1]  # Isto devia ser a última data com dados

        for key in data.keys():
            if data[key].index[-1] != t:
                raise Exception("Date is not the same in companies")

        self.history[t] = [input, results, dict(zip(list(self.wallet.keys()), [
            dict(zip(["Transaction", "Transaction_Amount", "Reward", "Reward_Check"], ["N", 0, 0, 1])) for i in
            range(len(self.wallet.keys()))]))]
        '''
        0 - Input 
            0 - Input LSTM para cada empresa ##ÍNDICE DE EMPRESA
                NpArray de tamanho #?eunaosei?#
            1 - Valores em lista da wallet ##ÍNDICE DE EMPRESA
        1 - Output sem rewards aplicados
        2 - Dicionários para cada empresa
            Transaction - Letra da transação (B, S, H, S*H, B*H)
            Transaction_Amount - Quantia da transação EM QUANTIDADE DE STOCK
            Reward - reward do Q value a ajustar
            Reward_Check - peso de reward a dar (B- = a numero de açoes compradas, S- É logo atualizado, outros binario)
        '''
        for i in range(len(results)):
            company = list(self.wallet.keys())[i]

            if np.any(results[i][0]) and np.random.random() > self.random_choice_chance:
                decision = np.argmax(results[i][0])
            else:
                decision = np.random.choice([1, 2, 0])
            softmax = lambda x, j: np.exp(x[j]) / sum(np.exp(x))

            if decision == 0:
                buy_price = self.get_buy_price(
                    data[company])
                amount = np.floor(softmax(results[i][0], 0) * self.balance * self.investible_fraction / buy_price)
                if self.balance - buy_price > 0:
                    amount = max(amount, 1)
                    self.buy(company, amount)
                    self.history[t][2][company]["Transaction"] = "B"
                    self.history[t][2][company]["Transaction_Amount"] = copy.deepcopy(amount)  # again, amount
                    self.history[t][2][company]["Reward_Check"] = copy.deepcopy(amount)  # again, amount
                else:
                    self.history[t][2][company]["Transaction"] = "B*H"
                    self.history[t][2][company]["Reward"] = copy.deepcopy(self.hold_reward)  # todo decidir se fica assim
                    self.history[t][2][company]["Reward_Check"] = 0
            elif decision == 1 and self.wallet[company][1] > 0:
                amount = np.floor(softmax(results[i][0], 1) * self.wallet[company][1])
                amount = max(amount, 1.0) #ATENÇÃO: SO É VÁLIDO SE COMPRAR EM INCREMENTOS DE 1 SENAO ELE VENDE MAIS DO QUE TEM
                self.sell(company, amount)
                self.history[t][2][company]["Transaction"] = "S"
                self.history[t][2][company]["Transaction_Amount"] = copy.deepcopy(amount)  # again, amount
                self.update_rewards(company, t)
            elif decision == 1 and self.wallet[company][1] == 0:  # aqui podera ser quando vende mais do que tem
                self.history[t][2][company]["Transaction"] = "S*H"
                self.history[t][2][company]["Reward"] = copy.deepcopy(self.hold_reward) # todo decidir se fica assim
                self.history[t][2][company]["Reward_Check"] = 0
            else:
                self.history[t][2][company]["Transaction"] = "H"
                self.history[t][2][company]["Reward"] = copy.deepcopy(self.hold_reward)  # todo decidir se fica assim
                self.history[t][2][company]["Reward_Check"] = 0

    def update_time(self, real_time=False, wait_fraction=0.2):
        """
        Atualiza o self.now

        :param real_time: Tenta sempre chegar ao ponto mais próximo do atual
        :param wait_fraction: Fração do tempo de
        período de trading que espera até haver dados novos, apenas quando em real time
        :return:
        """
        comp = yf.Ticker(list(self.wallet.keys())[-1])
        time_table = {'s': 1, 'm': 60, 'h': 3600, 'd': 3600 * 24}
        time_sleep_in_seconds = int(int(self.interval[:-1]) * time_table[self.interval[-1]])

        utc = pytz.UTC

        data = comp.history(start=self.now.strftime('%Y-%m-%d'),
                            end=(self.now + relativedelta(days=6)).strftime('%Y-%m-%d'), interval=self.interval)

        if not real_time:
            t = self.now + relativedelta(seconds=time_sleep_in_seconds)

            while t not in data.index:
                t += relativedelta(seconds=time_sleep_in_seconds)
                if t.replace(tzinfo=utc) > data.index[-1].replace(tzinfo=utc):
                    raise Exception("Hora dessincronizada: " + str(self.now) + " com hora maxima de resoluçao " + str(
                        data.index[-1]))
            self.now = t
        else:
            while data.index[-1] in self.history.keys():
                # Todo: Sim isto usa time.sleep e não sei se é muito bom
                time.sleep(time_sleep_in_seconds * wait_fraction)
                start = (datetime.datetime.now() - relativedelta(days=6))
                end = datetime.datetime.now()
                data = comp.history(start=start.strftime('%Y-%m-%d'), interval=self.interval,
                                    end=end.strftime('%Y-%m-%d')).tail(1)
            self.now = data.index[-1]

    def generate_historical_training_data(self, size=None, delete_history=False):
        """
        :param size: Output size. At default None will go as far as history can go for all stocks with Reward_Check = 0
        :param delete_history: Delete transactions transferred to output
        :return: input, output

        """
        output_values = {}
        lil_key = {"B": 0, "S": 1, "S*H": 2, "B*H":2, "H": 2}
        breh = 0
        for t in self.history.keys():
            if not np.any([self.history[t][2][i]["Reward_Check"] for i in self.wallet.keys()]):
                new_q_values = copy.deepcopy(self.history[t][1])
                for company in self.wallet.keys():
                    company_index = next(
                        (i for i in range(len(self.wallet.keys())) if list(self.wallet.keys())[i] == company))

                    try:
                        new_q_values[company_index][0][lil_key[self.history[t][2][company]["Transaction"]]] = max(
                            new_q_values[company_index][0][lil_key[self.history[t][2][company]["Transaction"]]] * (
                                        1 - self.q_learning_rate) + self.q_learning_rate * self.history[t][2][company][
                                "Reward"] + self.discount_factor*np.max(self.history[list(self.history.keys())[list(self.history.keys()).index(t)+1]][1][company_index]), 0)
                    except:
                        new_q_values[company_index][0][lil_key[self.history[t][2][company]["Transaction"]]] = max(
                            new_q_values[company_index][0][lil_key[self.history[t][2][company]["Transaction"]]] * (
                                        1 - self.q_learning_rate) + self.q_learning_rate * self.history[t][2][company][
                                "Reward"], 0)
                    # todo: Ver se fazemos reward negativo ao S*H e B*H
                output_values[t] = new_q_values
                breh += 1

            if size is not None and breh == size:
                break

        output = []
        input_lstm = [[] for i in range(len(self.wallet.keys()))]
        input_wallet = []
        for t in output_values.keys():
            output.append(output_values[t])
            for i in range(len(self.wallet.keys())):
                #input_lstm[i].append(self.history[t][0][i])
                input_lstm[i].append(np.array(self.history[t][0][i]))
            input_wallet.append(self.history[t][0][-1])
            # input.append(self.history[t][0])
            if delete_history:
                self.history.pop(t)


        '''
        concatenated_input = []
        for bre in input_lstm:
            concatenated_input.append(np.concatenate(bre))
        concatenated_input.append(np.concatenate(input_wallet))

        concatenated_lstm = np.concatenate(input_lstm)
        concatenated_wallet = np.concatenate(input_wallet)
        concatenated_input = [concatenated_lstm, concatenated_wallet]
        concatenated_output = np.concatenate(output)
        '''
        #return concatenated_input, concatenated_output

        #Todo: Ver se isto funciona afinal de tudo
        for i in range(len(input_lstm)):
            input_lstm[i] = np.concatenate(input_lstm[i])
        input_shalbong = [input_lstm] + [np.concatenate(input_wallet)]
        return input_shalbong, np.array(output)

    def check_history_for_trainable_data(self, size=60):
        alguma_coisa_mais_mamalhuda = 0

        for t in self.history.keys():
            if not np.any([self.history[t][2][i]["Reward_Check"] for i in self.wallet.keys()]):
                alguma_coisa_mais_mamalhuda += 1
            else:
                break
            if alguma_coisa_mais_mamalhuda == size:
                return True
        return False

    def update_rewards(self, company, update_time):
        """
        Modifies rewards in history
        ONLY AFTER A SALE
        :param company: Company which just sold
        :param update_time: Time at which the company's transaction was taken place
        :return:
        """
        if self.history[update_time][2][company]["Transaction"] != "S":
            raise Exception("Update_rewards mandado quando não venda")
        company_index = next((i for i in range(len(self.wallet.keys())) if list(self.wallet.keys())[i] == company))
        sell_price = self.get_sell_price(self.history[update_time][0][company_index], True)

        self.history[update_time][2][company]["Reward"] = ((sell_price - self.wallet[company][0])/sell_price)*self.history[update_time][2][company]["Transaction_Amount"]

        if self.verbose == 1:
            print((sell_price - self.wallet[company][0])/sell_price)
        self.history[update_time][2][company]["Reward_Check"] = 0
        reward_power = copy.deepcopy(self.history[update_time][2][company]["Transaction_Amount"])
        for t in self.history.keys():  # Temos de assumir que isto vai de mais velho pra mais novo
            if t - update_time < datetime.timedelta(0) and reward_power > 0:
                # BUY
                if self.history[t][2][company]["Transaction"] == "B" and self.history[t][2][company][
                    "Reward_Check"] > 0:
                    buy_price = self.get_buy_price(self.history[t][0][company_index], True)


                    #Reward de buy += fraçao de lucro * min(reward_power, amount)/amount
                    '''self.history[t][2][company]["Reward"] += ((sell_price - buy_price)/sell_price) * (
                        min(self.history[t][2][company]["Transaction_Amount"], reward_power)/self.history[t][2][company]["Transaction_Amount"]
                    )'''
                    #Nevermind, Reward += fraçao de lucro*min(reward_power, amount)
                    self.history[t][2][company]["Reward"] += ((sell_price - buy_price)/sell_price) *min(self.history[t][2][company]["Transaction_Amount"], reward_power)

                    reward_power_old = copy.deepcopy(reward_power)
                    reward_power = max(reward_power - self.history[t][2][company]["Reward_Check"], 0)
                    self.history[t][2][company]["Reward_Check"] = max(
                        self.history[t][2][company]["Reward_Check"] - reward_power_old, 0)
                # HOLD
                elif self.history[t][2][company]["Transaction"] == "H" and self.history[t][2][company][
                    "Reward_Check"] > 0:
                    self.history[t][2][company]["Reward"] = self.hold_reward
                    self.history[t][2][company]["Reward_Check"] = 0
                # SELL PROIBIDO FORÇADO A HOLD
                elif self.history[t][2][company]["Transaction"] == "S*H" and self.history[t][2][company][
                    "Reward_Check"] > 0:
                    self.history[t][2][company]["Reward"] = self.hold_reward
                    self.history[t][2][company]["Reward_Check"] = 0
                # BUY PROIBIDO FORÇADO A HOLD
                elif self.history[t][2][company]["Transaction"] == "B*H" and self.history[t][2][company][
                    "Reward_Check"] > 0:
                    self.history[t][2][company]["Reward"] = self.hold_reward
                    self.history[t][2][company]["Reward_Check"] = 0

    def train_model(self, size=None, epochs=5, delete_history=False):
        input_data, output_data = self.generate_historical_training_data(size=size, delete_history=delete_history)
        print(output_data)
        self.model.fit(input_data, output_data, batch_size=self.batch_size, epochs=epochs,
                       validation_split=self.validation_ratio, verbose=self.verbose)
        self.model.save(self.model_name)

    def create_model(self, stock_correlation_sizes=[1000, 500, 300, 100], wallet_correlation_sizes=[50, 30, 10],
                     prediction_sizes=[400, 200, 100], decision_sizes=[200, 100]):
        inputs = []
        stock_inputs = []
        input2 = Input(shape=(len(self.wallet.keys()), 2,))
        wallet_input = BatchNormalization(synchronized=True)(input2)

        for i in range(len(self.wallet)):
            inputs.append(Input(shape=(self.timesteps, 5,)))
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
                                                                                                i])
                # prediction_boys[i] = LSTM(units=prediction_sizes[0], input_shape=(self.timesteps, 5))(inputs[i])

                for size in prediction_sizes[1:-1]:
                    prediction_boys[i] = LSTM(units=size, return_sequences=True)(prediction_boys[i])
                prediction_boys[i] = LSTM(units=prediction_sizes[-1])(prediction_boys[i])

            # intermediate_layers = [Flatten()(prediction_boy) for prediction_boy in prediction_boys] + [ Flatten()(
            # wallet_boy)] + [ Flatten()(big_boy)]
            intermediate_layers = []
            for prediction_boy in prediction_boys:
                intermediate_layers.append([Flatten()(wallet_boy)] + [Flatten()(big_boy)] + [Flatten()(prediction_boy)])
            for layer in intermediate_layers:
                decision_boy = Concatenate(axis=1)(layer)
                for size in decision_sizes:
                    decision_boy = Dense(size, activation="relu")(decision_boy)
                decision_boy = Dense(3, activation="relu")(decision_boy)
                outputs.append(decision_boy)
        else:
            intermediate_layers = [Flatten()(wallet_boy)] + [Flatten()(big_boy)]
            decision_boy = Concatenate(axis=1)(intermediate_layers)
            for size in decision_sizes:
                decision_boy = Dense(size, activation="relu")(decision_boy)
            decision_boy = Dense(3, activation="relu")(decision_boy)
            outputs.append(decision_boy)

        self.model = Model(inputs=inputs + [input2], outputs=outputs)
        opt = Adam(learning_rate=self.learning_rate)

        self.model.compile(optimizer=opt, loss="mse")

    def create_model_tunable(self, hp):
        stock_correlation_sizes = [hp.Int("stock_correlation_"+str(i),min_value=3, max_value=2000)
                                   for i in range(hp.Int("stock_correlation_hidden_size",min_value=1, max_value=5))]
        wallet_correlation_sizes = [hp.Int("wallet_correlation_" + str(i), min_value=1, max_value=100)
                                    for i in
                                    range(hp.Int("wallet_correlation_hidden_size", min_value=1, max_value=5))]
        prediction_sizes = [hp.Int("prediction_"+str(i), min_value=5, max_value=1000)
                            for i in range(hp.Int("prediction_hidden_size", min_value=0, max_value=5))]

        decision_sizes = [hp.Int("decision_"+str(i), min_value=3, max_value=500)
                          for i in range(hp.Int("decision_hidden_size", min_value=1, max_value=5))]

        lr = hp.Float("learning_rate", min_value=0.00000001, max_value=0.1)

        self.batch_size = hp.Int("batch_size", min_value=2, max_value=32)
        inputs = []
        stock_inputs = []
        input2 = Input(shape=(len(self.wallet.keys()), 2,))
        wallet_input = BatchNormalization(synchronized=True)(input2)

        for i in range(len(self.wallet)):
            inputs.append(Input(shape=(self.timesteps, 5,)))
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
                                                                                                i])
                # prediction_boys[i] = LSTM(units=prediction_sizes[0], input_shape=(self.timesteps, 5))(inputs[i])

                for size in prediction_sizes[1:-1]:
                    prediction_boys[i] = LSTM(units=size, return_sequences=True)(prediction_boys[i])
                prediction_boys[i] = LSTM(units=prediction_sizes[-1])(prediction_boys[i])

            # intermediate_layers = [Flatten()(prediction_boy) for prediction_boy in prediction_boys] + [ Flatten()(
            # wallet_boy)] + [ Flatten()(big_boy)]
            intermediate_layers = []
            for prediction_boy in prediction_boys:
                intermediate_layers.append(
                    [Flatten()(wallet_boy)] + [Flatten()(big_boy)] + [Flatten()(prediction_boy)])
            for layer in intermediate_layers:
                decision_boy = Concatenate(axis=1)(layer)
                for size in decision_sizes:
                    decision_boy = Dense(size, activation="relu")(decision_boy)
                decision_boy = Dense(3, activation="relu")(decision_boy)
                outputs.append(decision_boy)
        else:
            intermediate_layers = [Flatten()(wallet_boy)] + [Flatten()(big_boy)]
            decision_boy = Concatenate(axis=1)(intermediate_layers)
            for size in decision_sizes:
                decision_boy = Dense(size, activation="relu")(decision_boy)
            decision_boy = Dense(3, activation="relu")(decision_boy)
            outputs.append(decision_boy)

        temp_model = Model(inputs=inputs + [input2], outputs=outputs)
        opt = Adam(learning_rate=lr)
        temp_model.compile(optimizer=opt, loss="mse")
        return temp_model

    def tune_model(self, tuner):
        X, Y = self.generate_historical_training_data(size=None, delete_history=False)
        '''
        X_val = X[int(len(X)*self.validation_ratio):]
        Y_val = Y[int(len(Y)*self.validation_ratio):]
        X = X[:int(len(X)*self.validation_ratio)]
        Y = Y[:int(len(Y)*self.validation_ratio)]
        tuner.search(X,Y, epochs=3, validation_data=(X_val, Y_val))
        '''
        tuner.search(X, Y, epochs=5, validation_split=self.validation_ratio)
        return tuner.get_best_models()

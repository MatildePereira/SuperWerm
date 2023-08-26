from trader import Trader
import numpy as np
import datetime
import tensorflow as tf
import random
from dateutil.relativedelta import relativedelta
import random
import pandas as pd
import keras_tuner
from keras.optimizers import Adam


# Press the green button in the gutter to run the script.


def create_trader():
    return Trader("James May", companies=["FDX", "EXPD", "HUBG"], init_balance=5000, buy_tax=0,
                  pessimism_factor=0, verbose=1)


if __name__ == '__main__':

    model_tuning = True
    trader = create_trader()
    trader.create_model(stock_correlation_sizes=[1000, 100], wallet_correlation_sizes=[50, 10],
                     prediction_sizes=[], decision_sizes=[100])

    trader.random_choice_chance = 1
    trader.now = pd.Timestamp('2021-' + str(random.randint(11, 12)) + '-' + str(
        random.randint(1, 29)) + ' 09:30:00-0400', tz='America/New_York')

    file = open("text.txt", "w")

    tuner = keras_tuner.BayesianOptimization(trader.create_model_tunable,
                                             objective='val_loss',
                                             max_trials=5)


    joe_biden = 0

    # trader.create_model()
    # trader.now = trader.get_stock_data()["INTC"].index[0] - relativedelta(days=60)
    # data = trader.get_stock_data(points=30)

    true_balance = trader.init_balance

    tf.keras.utils.plot_model(trader.model, to_file="model.png", show_shapes=True)

    while True:

        trader.decide_transaction()

        # We give the boy more mone
        trader.balance += 100
        joe_biden += 1

        # true_balance = hammond.balance -100
        # print(true_balance, hammond.wallet["FDX"][1] + hammond.wallet["EXPD"][1] + hammond.wallet["LSTR"][1])

        file.write((trader.now + relativedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S") + "\n")
        file.write("Balance: " + str(trader.balance) + "\n")
        file.write(np.array2string(np.array(list(trader.history.values())[-1][1])) + "\n")
        for company in list(trader.history.values())[-1][2].keys():
            file.write(list(trader.history.values())[-1][2][company]["Transaction"] + "-" + str(
                list(trader.history.values())[-1][2][company]["Transaction_Amount"]) + "\t")
        file.write("\n")

        file.write(np.array2string(np.array(trader.wallet.values())) + "\n")
        if joe_biden >= 180:
            trader.update_rewards()
            trader.random_choice_chance = max(trader.random_choice_chance - 0.05, 0.1)
            file.write("_*_*_*_*_*_*_*_TRAINED BOY_*_*_*_*_*_*_*_\n")

            # Para comparar loss tens de fazer history = model.fit() e depois history.history['loss'] ou 'val_loss' e isso retorna uma lista

            trader.model = trader.tune_model(tuner)[0]

            trader.train_model(None, delete_history=False)

            joe_biden = 0
        trader.update_time()
        file.write("****************************************************\n")

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

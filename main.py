from trader import Trader
import numpy as np
import datetime
import tensorflow as tf
import random
from dateutil.relativedelta import relativedelta
import random
import pandas as pd
# Press the green button in the gutter to run the script.


def create_trader():
    return Trader("James May", companies=["FDX", "EXPD", "HUBG"], init_balance=1000, hold_reward=0.05, buy_tax=0, pessimism_factor=0)


if __name__ == '__main__':

    model_tuning = False
    trader = create_trader()
    trader.random_choice_chance = 1
    trader.now = pd.Timestamp(str(random.randint(2021,2022))+'-'+str(random.randint(8,12))+'-'+str(random.randint(1,29))+ ' 09:30:00-0400', tz='America/New_York')

    file = open("text.txt", "w")

    #trader.create_model()
    #trader.now = trader.get_stock_data()["INTC"].index[0] - relativedelta(days=60)
    #data = trader.get_stock_data(points=30)

    true_balance = trader.init_balance

    while True:
        #hammond.decide_transaction()
        trader.decide_transaction()


        #We give the boy more mone
        trader.balance += 100
        #true_balance = hammond.balance -100
        #print(true_balance, hammond.wallet["FDX"][1] + hammond.wallet["EXPD"][1] + hammond.wallet["LSTR"][1])

        file.write((trader.now + relativedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S") + "\n")
        file.write("Balance: " + str(trader.balance) + "\n")
        file.write(np.array2string(np.array(list(trader.history.values())[-1][1])) + "\n")
        for company in list(trader.history.values())[-1][2].keys():
            file.write(list(trader.history.values())[-1][2][company]["Transaction"] +"-"+str(list(trader.history.values())[-1][2][company]["Transaction_Amount"]) + "\t")
        file.write("\n")

        file.write(np.array2string(np.array(trader.wallet.values())) + "\n")
        if trader.check_history_for_trainable_data():
            trader.train_model(None, delete_history= not model_tuning)
            trader.random_choice_chance = max(trader.random_choice_chance-0.05, 0.1)
            file.write("_*_*_*_*_*_*_*_TRAINED BOY_*_*_*_*_*_*_*_\n")
        trader.update_time()
        file.write("****************************************************\n")


    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

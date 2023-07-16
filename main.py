from trader import Trader
import numpy as np
import datetime
import tensorflow as tf
import random
from dateutil.relativedelta import relativedelta
import pandas as pd
# Press the green button in the gutter to run the script.


def create_trader():
    return Trader("Richard Hammond", companies=["FDX", "EXPD", "LSTR"], init_balance=1000)



if __name__ == '__main__':
    hammond = create_trader()
    file = open("text.txt", "w")

    #trader.create_model()
    #trader.now = trader.get_stock_data()["INTC"].index[0] - relativedelta(days=60)
    #data = trader.get_stock_data(points=30)

    for i in range(500):
        hammond.decide_transaction()
        file.write((hammond.now + relativedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"))
        file.write(hammond.history.values()[-1][1])
        file.write(hammond.wallet)
        if hammond.check_history_for_trainable_data():
            hammond.train_model(60)
            file.write("#######HAMMOOOOND#######")
        hammond.update_time()
        file.write("****************************************************")

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

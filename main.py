from trader import Trader
import numpy as np
import datetime
import tensorflow as tf
import random
from dateutil.relativedelta import relativedelta
import pandas as pd
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trader = Trader(None, companies=["INTC", "GOOG"])
    trader.now = datetime.date(2023,6,20)
    data = trader.get_stock_data(immediately=False, points=3)
    trader.create_model()
    for i in range(3):
        trader.decide_transaction()
        print(trader.now)
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

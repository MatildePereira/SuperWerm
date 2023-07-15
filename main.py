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
    #trader.create_model()
    #trader.now = trader.get_stock_data(immediately=False)["INTC"].index[0] - relativedelta(days=60)
    data = trader.get_stock_data(points=30)



    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

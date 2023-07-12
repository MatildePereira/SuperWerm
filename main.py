from trader import Trader
import numpy as np
import datetime
import tensorflow as tf

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trader = Trader(None, companies=["INTC", "GOOG"])
    data = trader.get_stock_data("1mo")
    trader.create_model()
    input = trader.prepare_stock_data(data, 1)
    print(input)
    print(trader.model.predict(input))

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

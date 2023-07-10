from trader import Trader
import tensorflow as tf

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trader = Trader(None, companies=["INTC", "GOOG"])
    data = trader.get_stock_data("1mo")
    print(data["INTC"][['Open', 'High', 'Low', 'Close', 'Volume']].tail())
    print(data["INTC"][['Open', 'High', 'Low', 'Close', 'Volume']].tail(1))

    print(list(trader.prepare_stock_data(data).values())[0][0])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

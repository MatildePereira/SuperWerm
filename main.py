from trader import Trader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trader = Trader(None)
    print(trader.get_stock_data("GOOG", "1d").to_string())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

from trader import Trader
import tensorflow as tf

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trader = Trader(None, companies=["GOOG", "AAPL", "MSFT"])
    #print(trader.get_stock_data('AAPL', period="1mo")[['Open', 'High', 'Low','Close','Volume']])
    trader.create_model()
    tf.keras.utils.plot_model (trader.model, show_shapes=True)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

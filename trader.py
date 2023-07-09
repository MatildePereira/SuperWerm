import yfinance as yf


class Trader:
    def __init__(self, model, init_balance=100, balance=100, stock_history=[], companies=['GOOG', 'APPL'],
                 interval="1h", buy_tax=0.06):
        self.init_balance = init_balance
        self.balance = balance
        self.model = model
        self.stock_history = stock_history
        self.company_amount = len(companies)
        self.wallet = {}
        for i in companies:
            self.wallet[i] = [None, 0]
        self.interval = interval
        self.buy_tax = buy_tax

    def buy(self, company, amount):
        """
        :param company: indicator of which company you're buying from
        :param amount: quantity of stocks to buy
        :void
        """
        self.wallet[company][1] += amount

    def sell(self, company, amount):
        """
        :param company: indicator of which company you're selling the stock
        :param amount: quantity of stocks to sell
        :void
        """
        self.wallet[company][1] -= amount

    def get_stock_data(self, company, period="immediate"):
        comp = yf.Ticker(company)
        if (period != "immediate"):
            hist = comp.history(period=period, interval=self.interval)
        else:
            hist = comp.history(period="1d", interval="1m").values[-1]
        return hist

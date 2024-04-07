class Generator():
    def __init__(self):
        pass

        """
        SMA(self, data, windows): Calculates the Simple Moving Average (SMA), which is the average stock price over a specific window of time.
        """

    def SMA(self, data, windows):
        res = data.rolling(window = windows).mean()
        return res
    
    """
    EMA(self, data, windows): Computes the Exponential Moving Average (EMA), giving more weight to recent prices and thus 
    reacting more significantly to recent price changes than the SMA.
    """

    def EMA(self, data, windows):
        res = data.ewm(span = windows).mean()
        return res

    """
    MACD(self, data, long, short, windows): The Moving Average Convergence Divergence (MACD) is a trend-following 
    momentum indicator that shows the relationship between two EMAs of prices. 
    """

    def MACD(self, data, long, short, windows):
        short_ = data.ewm(span = short).mean()
        long_ = data.ewm(span = long).mean()
        macd_ = short_ - long_
        res = macd_.ewm(span = windows).mean()
        return res
    
    """
    RSI(self, data, windows): The Relative Strength Index (RSI) measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
    """

    def RSI(self, data, windows):
        delta = data.diff(1)
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_up = up.rolling(window = windows).mean()
        avg_down = down.rolling(window = windows).mean()
        rs = avg_up/ avg_down
        rsi = 100. -(100./ (1. + rs))
        return rsi
    
    """
    atr(self, data_high, data_low, windows): The Average True Range (ATR) is an indicator that measures market 
    volatility by decomposing the entire range of an asset price for that period.
    """

    def atr(self, data_high, data_low, windows):
        range_ = data_high - data_low
        res = range_.rolling(window = windows).mean()
        return res

    """
    bollinger_band(self, data, windows): Bollinger Bands consist of an SMA (middle band) and two standard deviation lines (bands) 
    above and below the SMA. They help in identifying the volatility and overbought or oversold conditions in the price of an asset.
    """

    def bollinger_band(self, data, windows):
        sma = data.rolling(window = windows).mean()
        std = data.rolling(window = windows).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, lower
    
    """
    rsv(self, data, windows): The Raw Stochastic Value (RSV) is used to calculate the Stochastic Oscillator, 
    which is a momentum indicator comparing a particular closing price of an asset to a range of its prices over a certain period of time.
    """

    def rsv(self, data, windows):
        min_ = data.rolling(window = windows).min()
        max_ = data.rolling(window = windows).max()
        res = (data - min_)/ (max_ - min_) * 100
        return res

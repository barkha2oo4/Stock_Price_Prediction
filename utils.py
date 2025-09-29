# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fetch_data(ticker: str, start: str = "2015-01-01", end: str = None):
    """Download OHLCV from yfinance and return dataframe with Date index."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Open','High','Low','Close','Volume']]
    df.index = pd.to_datetime(df.index)
    return df.dropna()


def create_technical_features(df: pd.DataFrame):
    d = df.copy()
    d['return_1d'] = d['Close'].pct_change()
    d['ma_7'] = d['Close'].rolling(7).mean()
    d['ma_21'] = d['Close'].rolling(21).mean()
    d['std_7'] = d['Close'].rolling(7).std()
    d['vol_7'] = d['Volume'].rolling(7).mean()
    d['dayofweek'] = d.index.dayofweek

    # Add more technical indicators using ta
    try:
        import ta
        close = d['Close'].squeeze()
        high = d['High'].squeeze()
        low = d['Low'].squeeze()
        volume = d['Volume'].squeeze()
        d['rsi_14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        macd = ta.trend.MACD(close=close)
        d['macd'] = macd.macd()
        d['macd_signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(close=close)
        d['bb_bbm'] = bb.bollinger_mavg()
        d['bb_bbh'] = bb.bollinger_hband()
        d['bb_bbl'] = bb.bollinger_lband()
        d['ema_10'] = ta.trend.EMAIndicator(close=close, window=10).ema_indicator()
        d['ema_21'] = ta.trend.EMAIndicator(close=close, window=21).ema_indicator()
        d['willr'] = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r()
        d['adx'] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
    except ImportError:
        pass

    d = d.dropna()
    return d


def create_lag_features(df: pd.DataFrame, lags=[1,2,3,5,7,14]):
    d = df.copy()
    for l in lags:
        d[f'lag_{l}'] = d['Close'].shift(l)
    d = d.dropna()
    return d


def create_sequences(series_values: np.ndarray, seq_len: int = 30):
    X, y = [], []
    for i in range(seq_len, len(series_values)):
        X.append(series_values[i-seq_len:i, 0])
        y.append(series_values[i, 0])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)
    return X, y


def scale_series(series: pd.DataFrame, scaler: MinMaxScaler = None):
    s = series.values.reshape(-1,1)
    if scaler is None:
        scaler = MinMaxScaler()
        s_scaled = scaler.fit_transform(s)
    else:
        s_scaled = scaler.transform(s)
    return s_scaled, scaler

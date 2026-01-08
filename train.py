# train.py

import os
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

from utils import fetch_data, create_technical_features, create_lag_features, create_sequences, scale_series

# ------------------ Utils ------------------ #
def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


# ------------------ XGBoost Training ------------------ #
def train_xgboost(df_features, output_dir, ticker):
    stock_dir = os.path.join(output_dir, ticker)
    os.makedirs(stock_dir, exist_ok=True)

    df = df_features.dropna().copy()
    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    FEATURES = [c for c in df.columns if c != 'Close']
    X = df[FEATURES].values
    y = df['Close'].values

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Tuned hyperparameters
    model = XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    try:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False)
    except TypeError:
        # Fallback for older xgboost versions
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    print("XGBoost RMSE:", rmse(y_test, preds))

    # Save model and feature columns
    joblib.dump(model, os.path.join(stock_dir, 'xgb_model.joblib'))
    joblib.dump(FEATURES, os.path.join(stock_dir, 'feature_cols.joblib'))
    print(f'Saved xgb_model.joblib and feature_cols.joblib for {ticker}')


# ------------------ LSTM Training ------------------ #
def train_lstm(series_df, output_dir, ticker, seq_len=30):
    stock_dir = os.path.join(output_dir, ticker)
    os.makedirs(stock_dir, exist_ok=True)
    
    s = series_df[['Close']].copy()
    scaled, scaler = scale_series(s)

    X, y = create_sequences(scaled, seq_len=seq_len)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(128, input_shape=(seq_len,1), return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
    epochs=100, batch_size=32, callbacks=[es], verbose='auto')

    preds = model.predict(X_test).ravel()
    preds_rescaled = scaler.inverse_transform(preds.reshape(-1,1)).ravel()
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
    print('LSTM RMSE:', rmse(y_test_rescaled, preds_rescaled))

    # Save model and scaler
    model.save(os.path.join(stock_dir, 'lstm_model.keras'))
    joblib.dump(scaler, os.path.join(stock_dir, 'scaler.joblib'))
    print(f'Saved lstm_model.keras and scaler.joblib for {ticker}')


# ------------------ Main ------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--out', default='models')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f'Fetching data for {args.ticker}')
    df = fetch_data(args.ticker, start=args.start)

    # Create features
    if df is not None:
        tech = create_technical_features(df)
        df_lags = create_lag_features(tech)

        # Train XGBoost
        train_xgboost(df_lags, args.out, args.ticker)

    # Train LSTM
    train_lstm(tech[['Close']], args.out, args.ticker, seq_len=30)

    print(f'All models trained and saved to {os.path.join(args.out, args.ticker)}')
# ------------------ End ------------------ #
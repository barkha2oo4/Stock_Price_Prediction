# app.py

import os
import joblib
import json
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
try:
    from keras.models import load_model
except ImportError:
    load_model = None
from utils import fetch_data, create_technical_features, create_lag_features, create_sequences

app = Flask(__name__)
CORS(app)
MODEL_DIR = os.environ.get('MODEL_DIR', 'models')

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Global variables
models = {}  # Dictionary to store models for each stock

# ------------------ Load Models ------------------ #
def load_stock_models(ticker):
    if ticker in models:
        return models[ticker]
    
    stock_dir = os.path.join(MODEL_DIR, ticker)
    if not os.path.exists(stock_dir):
        return None

    try:
        xgb_path = os.path.join(stock_dir, 'xgb_model.joblib')
        lstm_path = os.path.join(stock_dir, 'lstm_model.keras')
        scaler_path = os.path.join(stock_dir, 'scaler.joblib')
        feature_cols_path = os.path.join(stock_dir, 'feature_cols.joblib')

        stock_models = {}
        
        if os.path.exists(xgb_path):
            stock_models['xgb_model'] = joblib.load(xgb_path)
            logging.info(f'Loaded XGBoost model for {ticker}.')

        if os.path.exists(feature_cols_path):
            stock_models['feature_cols'] = joblib.load(feature_cols_path)
            logging.info(f'Loaded XGBoost feature columns for {ticker}.')

        if os.path.exists(lstm_path) and load_model:
            stock_models['lstm_model'] = load_model(lstm_path)
            logging.info(f'Loaded LSTM model for {ticker}.')

        if os.path.exists(scaler_path):
            stock_models['scaler'] = joblib.load(scaler_path)
            logging.info(f'Loaded scaler for {ticker}.')
            
        if stock_models:
            models[ticker] = stock_models
            return stock_models
    except Exception as e:
        logging.error(f'Error loading models: {e}')


# ------------------ Endpoints ------------------ #
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'loaded_models': list(models.keys())
    })


@app.route('/version')
def version():
    import sys
    import xgboost, tensorflow, pkg_resources
    flask_version = pkg_resources.get_distribution('flask').version
    return jsonify({
        'python': sys.version,
        'xgboost': xgboost.__version__,
        'tensorflow': tensorflow.__version__,
        'flask': flask_version
    })


@app.route('/history')
def history():
    ticker = request.args.get('ticker', 'AAPL')
    try:
        df = fetch_data(ticker)
        df = df.tail(30)
        dates = [d.strftime('%Y-%m-%d') for d in df.index]
        closes = df['Close'].values.tolist()
        opens = df['Open'].values.tolist()
        highs = df['High'].values.tolist()
        lows = df['Low'].values.tolist()
        volumes = df['Volume'].values.tolist()
        return jsonify({
            'dates': dates,
            'closes': closes,
            'opens': opens,
            'highs': highs,
            'lows': lows,
            'volumes': volumes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json()
        model_choice = payload.get('model', 'xgb')
        ticker = payload.get('ticker', 'AAPL')

        # Load models for the specified ticker
        stock_models = load_stock_models(ticker)
        if stock_models is None:
            return jsonify({'error': f'No models found for ticker {ticker}'}), 404

        # Get data
        if 'data' in payload:
            df = pd.DataFrame(payload['data'])
            if 'index' in df.columns:
                df.set_index(pd.to_datetime(df['index']), inplace=True)
        else:
            df = fetch_data(ticker)

        # Feature engineering
        tech = create_technical_features(df)
        df_lags = create_lag_features(tech)
        if df_lags.shape[0] == 0:
            return jsonify({'error': 'Not enough data to create lag features.'}), 400

        last_row = df_lags.iloc[-1:]

        # ---------- XGBoost Prediction ---------- #
        if model_choice == 'xgb':
            if 'xgb_model' not in stock_models or 'feature_cols' not in stock_models:
                return jsonify({'error': 'XGBoost model not loaded.'}), 500
            try:
                X = last_row[stock_models['feature_cols']].values
                pred = stock_models['xgb_model'].predict(X)[0]
                return jsonify({'model':'xgb', 'prediction': float(pred), 'ticker': ticker})
            except Exception as e:
                return jsonify({'error': f'XGBoost prediction failed: {e}'}), 500

        # ---------- LSTM Prediction ---------- #
        elif model_choice == 'lstm':
            if 'lstm_model' not in stock_models or 'scaler' not in stock_models:
                return jsonify({'error': 'LSTM model or scaler not loaded.'}), 500
            seq_len = stock_models['lstm_model'].input_shape[1]
            series = tech[['Close']].copy()
            scaled = stock_models['scaler'].transform(series.values.reshape(-1,1))
            if len(scaled) < seq_len:
                return jsonify({'error': f'Not enough data for LSTM sequence. Require at least {seq_len} rows.'}), 400
            seq = scaled[-seq_len:].reshape(1, seq_len, 1)
            pred_scaled = stock_models['lstm_model'].predict(seq, verbose=0).ravel()[0]
            pred = stock_models['scaler'].inverse_transform([[pred_scaled]])[0][0]
            return jsonify({'model':'lstm', 'prediction': float(pred), 'ticker': ticker})

        else:
            return jsonify({'error': 'Invalid model choice. Choose "xgb" or "lstm".'}), 400

    except Exception as e:
        logging.error(f'/predict error: {e}')
        return jsonify({'error': str(e)}), 500


# ------------------ Main ------------------ #
if __name__ == '__main__':
    # Models will be loaded on-demand for each ticker
    app.run(host='0.0.0.0', port=5000, debug=True)

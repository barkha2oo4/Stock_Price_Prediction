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
xgb_model = None
lstm_model = None
scaler = None
feature_cols = None  # XGBoost feature columns

# ------------------ Load Models ------------------ #
def load_models():
    global xgb_model, lstm_model, scaler, feature_cols
    try:
        xgb_path = os.path.join(MODEL_DIR, 'xgb_model.joblib')
        lstm_path = os.path.join(MODEL_DIR, 'lstm_model.keras')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
        feature_cols_path = os.path.join(MODEL_DIR, 'feature_cols.joblib')

        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            logging.info('Loaded XGBoost model.')

        if os.path.exists(feature_cols_path):
            feature_cols = joblib.load(feature_cols_path)
            logging.info('Loaded XGBoost feature columns.')

        if os.path.exists(lstm_path) and load_model:
            lstm_model = load_model(lstm_path)
            logging.info('Loaded LSTM model.')

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info('Loaded scaler.')
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
        'xgb_model': xgb_model is not None,
        'lstm_model': lstm_model is not None,
        'scaler': scaler is not None
    })


@app.route('/version')
def version():
    import sys
    import xgboost, tensorflow, flask
    return jsonify({
        'python': sys.version,
        'xgboost': xgboost.__version__,
        'tensorflow': tensorflow.__version__,
        'flask': flask.__version__
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

        # Get data
        if 'data' in payload:
            df = pd.DataFrame(payload['data'])
            if 'index' in df.columns:
                df.index = pd.to_datetime(df['index'])
        else:
            ticker = payload.get('ticker', 'AAPL')
            df = fetch_data(ticker)

        # Feature engineering
        tech = create_technical_features(df)
        df_lags = create_lag_features(tech)
        if df_lags.shape[0] == 0:
            return jsonify({'error': 'Not enough data to create lag features.'}), 400

        last_row = df_lags.iloc[-1:]

        # ---------- XGBoost Prediction ---------- #
        if model_choice == 'xgb':
            if xgb_model is None or feature_cols is None:
                return jsonify({'error': 'XGBoost model not loaded.'}), 500
            try:
                X = last_row[feature_cols].values  # ensure correct order
                pred = xgb_model.predict(X)[0]
                return jsonify({'model':'xgb', 'prediction': float(pred)})
            except Exception as e:
                return jsonify({'error': f'XGBoost prediction failed: {e}'}), 500

        # ---------- LSTM Prediction ---------- #
        elif model_choice == 'lstm':
            if lstm_model is None or scaler is None:
                return jsonify({'error': 'LSTM model or scaler not loaded.'}), 500
            seq_len = lstm_model.input_shape[1]
            series = tech[['Close']].copy()
            scaled = scaler.transform(series.values.reshape(-1,1))
            if len(scaled) < seq_len:
                return jsonify({'error': f'Not enough data for LSTM sequence. Require at least {seq_len} rows.'}), 400
            seq = scaled[-seq_len:].reshape(1, seq_len, 1)
            pred_scaled = lstm_model.predict(seq, verbose=0).ravel()[0]
            pred = scaler.inverse_transform([[pred_scaled]])[0][0]
            return jsonify({'model':'lstm', 'prediction': float(pred)})

        else:
            return jsonify({'error': 'Invalid model choice. Choose "xgb" or "lstm".'}), 400

    except Exception as e:
        logging.error(f'/predict error: {e}')
        return jsonify({'error': str(e)}), 500


# ------------------ Main ------------------ #
if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)

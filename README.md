
# Advanced Stock Price Prediction Pipeline (Flask, XGBoost, LSTM)

## Features
- End-to-end pipeline: data fetch, feature engineering, XGBoost & LSTM training
- Modern Flask API with `/predict`, `/health`, `/version`, `/history` endpoints
- Responsive Bootstrap 5 web UI with chart and error handling
- Dockerized for production
- Logging and error handling

## Quickstart

1. **Install dependencies**
	```sh
	pip install -r requirements.txt
	```
2. **Train models**
	```sh
	python train.py --ticker AAPL --start 2015-01-01 --out models
	```
3. **Run the API**
	```sh
	python app.py
	# or for production
	gunicorn -w 2 -b 0.0.0.0:5000 app:app
	```
4. **Open the UI**
	- Go to [http://localhost:5000](http://localhost:5000)

## Docker
```sh
# Build
docker build -t stock-predictor .
# Run
docker run -p 5000:5000 stock-predictor
```

## Endpoints
- `/predict` (POST): Predict next close price
- `/health`: Health check
- `/version`: Python, XGBoost, TensorFlow, Flask versions
- `/history?ticker=...`: Last 30 closes for chart

## Professional Tips
- Use `.env` for secrets/config (see `.env.example`)
- Add monitoring, authentication, and CI/CD for production
- Retrain models regularly for best results

## License
MIT

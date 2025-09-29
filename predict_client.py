# predict_client.py
import requests
url = 'http://localhost:5000/predict'
payload = {'ticker':'AAPL', 'model':'xgb'}
resp = requests.post(url, json=payload)
print(resp.json())

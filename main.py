import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
timeframe = '1d'

@lru_cache(maxsize=None)
def fetch_stock_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

def calculate_SMA(data, period):
    return data.rolling(window=period).mean()

def calculate_EMA(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_RSI(data, period):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_BollingerBands(data, period):
    sma = calculate_SMA(data, period)
    std_dev = data.rolling(window=period).std()
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    return upper_band, lower_band

def preprocess_data(data):
    ml_data = pd.DataFrame()

    for symbol, df in data.items():
        df = df.copy()
        df['symbol'] = symbol
        ml_data = pd.concat([ml_data, df], ignore_index=True)


    ml_data.dropna(inplace=True)
    features = ml_data.drop(columns=['symbol', 'Close']).values
    targets = (ml_data['Close'].pct_change() > 0).astype(int).values

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    return {'X': features, 'y': targets}

def generate_signals(model, data):
    signals = {}

    for symbol, df in data.items():
        latest_data = df.iloc[-1].drop(labels=['Close'])
        latest_features = latest_data.values.reshape(1, -1)

        signal = model.predict(latest_features)[0]
        signals[symbol] = 'Buy' if signal == 1 else 'Sell'

    return signals

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

stock_data = {}
for symbol in symbols:
    stock_data[symbol] = fetch_stock_data(symbol, start_date, end_date)

for symbol, data in stock_data.items():
    data['SMA'] = calculate_SMA(data['Close'], 14)
    data['EMA'] = calculate_EMA(data['Close'], 14)
    data['RSI'] = calculate_RSI(data['Close'], 14)
    data['UpperBB'], data['LowerBB'] = calculate_BollingerBands(data['Close'], 20)

prepared_data = preprocess_data(stock_data)

X_train, X_test, y_train, y_test = train_test_split(prepared_data['X'], prepared_data['y'], test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")

signals = generate_signals(model, stock_data)
print(signals)

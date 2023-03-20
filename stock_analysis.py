import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta import add_all_ta_features
from ta.utils import dropna
import matplotlib.pyplot as plt


symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN',"SIE.DE"]
timeframe = '1d'




@lru_cache(maxsize=None) # Cache the results of the function
def fetch_stock_data(symbol, start, end): # Fetch the stock data from Yahoo Finance
    data = yf.download(symbol, start=start, end=end) # Download the data
    return data # Return the data

def calculate_SMA(data, period): # Calculate the Simple Moving Average
    return data.rolling(window=period).mean() # Return the SMA

def calculate_EMA(data, period): # Calculate the Exponential Moving Average
    return data.ewm(span=period, adjust=False).mean() # Return the EMA

def calculate_MACD(data, short_period, long_period, signal_period): # Calculate the Moving Average Convergence Divergence 
    short_EMA = data.ewm(span=short_period, adjust=False).mean() # Calculate the short term exponential moving average
    long_EMA = data.ewm(span=long_period, adjust=False).mean () # Calculate the long term exponential moving average
    MACD = short_EMA - long_EMA # Calculate the MACD
    MACD_signal = MACD.ewm(span=signal_period, adjust=False).mean() # Calculate the signal line
    return MACD, MACD_signal # Return the MACD and signal line

def calculate_ADX(data, period): # Calculate the Average Directional Movement Index
    adx = ADXIndicator(data['High'], data['Low'], data['Close'], period) # Create the ADX indicator
    return adx.adx() # Return the ADX value

def calculate_Stochastic_Oscillator(data, k_period, d_period): # Calculate the Stochastic Oscillator
    stoch = StochasticOscillator(data['High'], data['Low'], data['Close'], k_period) # Create the Stochastic Oscillator
    return stoch.stoch(), stoch.stoch_signal() # Return the Stochastic Oscillator and signal line

def calculate_RSI(data, period): # Calculate the Relative Strength Index
    delta = data.diff() # Calculate the difference between the current and previous price
    gain = delta.where(delta > 0, 0) # Calculate the gain
    loss = -delta.where(delta < 0, 0) # Calculate the loss

    avg_gain = gain.rolling(window=period).mean() # Calculate the average gain
    avg_loss = loss.rolling(window=period).mean() # Calculate the average loss

    rs = avg_gain / avg_loss # Calculate the Relative Strength
    rsi = 100 - (100 / (1 + rs)) # Calculate the Relative Strength Index
    return rsi # Return the RSI value

def calculate_BollingerBands(data, period): # Calculate the Bollinger Bands
    sma = calculate_SMA(data, period) # Calculate the Simple Moving Average
    std_dev = data.rolling(window=period).std() # Calculate the standard deviation
    upper_band = sma + (2 * std_dev) # Calculate the upper band
    lower_band = sma - (2 * std_dev) # Calculate the lower band
    return upper_band, lower_band # Return the upper and lower bands

def preprocess_data(data): # Preprocess the data for the machine learning model
    ml_data = pd.DataFrame() # Create an empty DataFrame

    for symbol, df in data.items(): # Loop through the data
        df = df.copy() # Copy the DataFrame for manipulation
        df['symbol'] = symbol # Add a column for the symbol
        ml_data = pd.concat([ml_data, df], ignore_index=True) # Add the data to the DataFrame


    ml_data.dropna(inplace=True) # Drop the missing values
    features = ml_data.drop(columns=['symbol', 'Close']).values # Get the features
    targets = (ml_data['Close'].pct_change() > 0).astype(int).values # Get the targets

    scaler = MinMaxScaler() # Create a scaler
    features = scaler.fit_transform(features) # Scale the features

    return {'X': features, 'y': targets} # Return the features and targets

def generate_signals(model, data, prediction_date): # Generate the trading signals
    signals = {} # Create an empty dictionary

    for symbol, df in data.items(): # Loop through the data
        latest_data = df.loc[df.index < prediction_date].iloc[-1].drop(labels=['Close']) # Get the latest data before the prediction date
        latest_features = latest_data.values.reshape(1, -1) # Reshape the data

        signal = model.predict(latest_features)[0] # Get the signal
        signals[symbol] = 'Buy' if signal == 1 else 'Sell' # Add the signal to the dictionary

    return signals # Return the signals


def get_prediction_date(end_date, trading_frequency):  # Get the prediction date
    if trading_frequency == 'daily':  # Add 1 day for daily trading
        prediction_date = end_date + timedelta(days=1)  # Add 1 day
    elif trading_frequency == 'weekly':  # Add 1 week for weekly trading
        prediction_date = end_date + timedelta(weeks=1)  # Add 1 week
    elif trading_frequency == 'monthly':  # Add 1 month for monthly trading
        prediction_date = end_date + timedelta(days=30)  # Approximately 1 month
    else:
        raise ValueError("Invalid trading frequency. Choose 'daily', 'weekly', or 'monthly'.")
    return prediction_date

print("Choose your trading frequency:")
print("1. Daily")
print("2. Weekly")
print("3. Monthly")
user_choice = int(input("Enter the number (1, 2, or 3): "))

if user_choice == 1:
    trading_frequency = 'daily'
elif user_choice == 2:
    trading_frequency = 'weekly'
elif user_choice == 3:
    trading_frequency = 'monthly'
else:
    raise ValueError("Invalid choice. Choose 1, 2, or 3.")

end_date = datetime.now()  # Get the current date and time
start_date = end_date - timedelta(days=365)  # Get the start date 1 year ago
prediction_date = get_prediction_date(end_date, trading_frequency)

def get_prediction_date(end_date, trading_frequency): # Get the prediction date
    if trading_frequency == 'daily': # Add 1 day for daily trading
        prediction_date = end_date + timedelta(days=1) # Add 1 day
    elif trading_frequency == 'weekly': # Add 1 week for weekly trading
        prediction_date = end_date + timedelta(weeks=1) # Add 1 week
    elif trading_frequency == 'monthly': # Add 1 month for monthly trading
        prediction_date = end_date + timedelta(days=30) # Approximately 1 month
    else:
        raise ValueError("Invalid trading frequency. Choose 'daily', 'weekly', or 'monthly'.")
    return prediction_date

stock_data = {} # Create an empty dictionary
for symbol in symbols:  # Loop through the symbols
    stock_data[symbol] = fetch_stock_data(symbol, start_date, end_date) # Fetch the stock data

for symbol, data in stock_data.items(): # Loop through the data
    data['SMA'] = calculate_SMA(data['Close'], 14) # Calculate the Simple Moving Average for 14 days
    data['EMA'] = calculate_EMA(data['Close'], 14) # Calculate the Exponential Moving Average for 14 days
    data['RSI'] = calculate_RSI(data['Close'], 14) # Calculate the Relative Strength Index for 14 days
    data['MACD'], data['MACD_Signal'] = calculate_MACD(data['Close'], 12, 26, 9)
    data['ADX'] = calculate_ADX(data, 14) # Calculate the Average Directional Movement Index for 14 days
    data['Stochastic_Oscillator'], data['Stochastic_Signal'] = calculate_Stochastic_Oscillator(data, 14, 3) # Calculate the Stochastic Oscillator for 14 days
    data['UpperBB'], data['LowerBB'] = calculate_BollingerBands(data['Close'], 20) # Calculate the Bollinger Bands for 20 days

prepared_data = preprocess_data(stock_data) # Preprocess the data

X_train, X_test, y_train, y_test = train_test_split(prepared_data['X'], prepared_data['y'], test_size=0.2, random_state=42) # Split the data into training and testing sets

model = RandomForestClassifier(random_state=42) # Create the model
model.fit(X_train, y_train) # Train the model

y_pred = model.predict(X_test) # Make predictions
accuracy = accuracy_score(y_test, y_pred) # Calculate the accuracy


# Prepare the data for LSTM model
def create_dataset(X, y, time_steps=1):  # Create the dataset
    Xs, ys = [], []  # Create empty lists
    for i in range(len(X) - time_steps):  # Loop through the data
        v = X.iloc[i:(i + time_steps)].values  # Get the values
        Xs.append(v)  # Add the values to the list
        ys.append(y.iloc[i + time_steps])  # Add the target to the list
    return np.array(Xs), np.array(ys)  # Return the data and targets

# LSTM model parameters
time_steps = 10  # Number of previous days to use for prediction
batch_size = 32  # Number of samples per gradient update
epochs = 10  # Number of epochs to train the model

# Prepare the data
X = pd.DataFrame(prepared_data['X'])  # Get the features
y = pd.Series(prepared_data['y'])  # Get the targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data into training and testing sets

# Reshape the data into sequences for LSTM
X_train_lstm, y_train_lstm = create_dataset(X_train, y_train, time_steps)
X_test_lstm, y_test_lstm = create_dataset(X_test, y_test, time_steps)

# Define the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the LSTM model
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the LSTM model
lstm_accuracy = model_lstm.evaluate(X_test_lstm, y_test_lstm)[1]


def plot_stock_data(stock_data, symbol, start_date, end_date): # Plot the stock data
    data = stock_data[symbol]
    data = data.loc[start_date:end_date]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12, 12))
    ax1.plot(data.index, data['Close'], label='Close Price')
    ax1.plot(data.index, data['SMA'], label='SMA (14 days)')
    ax1.plot(data.index, data['EMA'], label='EMA (14 days)')
    ax1.set_title(f'{symbol} Stock Price')
    ax1.legend(loc='upper left')
    ax2.plot(data.index, data['RSI'], label='RSI (14 days)')
    ax2.axhline(30, color='red', linestyle='--')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.set_title('Relative Strength Index')
    ax2.legend(loc='upper left')
    ax3.plot(data.index, data['MACD'], label='MACD (12, 26, 9)')
    ax3.plot(data.index, data['MACD_Signal'], label='Signal Line')
    ax3.set_title('Moving Average Convergence Divergence')
    ax3.legend(loc='upper left')
    ax4.plot(data.index, data['Stochastic_Oscillator'], label='Stochastic Oscillator')
    ax4.plot(data.index, data['Stochastic_Signal'], label='Signal Line')
    ax4.set_title('Stochastic Oscillator')
    ax4.legend(loc='upper left')
    plt.tight_layout()
    plt.show()




print(f"LSTM Model accuracy: {lstm_accuracy:.2f}")
print(f"Random Forest Model accuracy: {accuracy:.2f}")

signals = generate_signals(model, stock_data, prediction_date)
print(signals)

plot_stock_data(stock_data, 'AMZN', start_date, end_date)







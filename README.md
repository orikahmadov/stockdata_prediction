Dependencies
Python 3.6+
pandas
numpy
yfinance
scikit-learn
tensorflow

You can install the required libraries using pip or pipenv:
pip install -r requirements.txt

Open a terminal or command prompt and navigate to the directory where the script is located.

Run the script using the following command:

(venv) python stock_analysis.py
The script will execute the following steps:

Fetch stock data for the specified symbols (AAPL, GOOGL, TSLA, MSFT, AMZN) for the past year.
Calculate technical indicators: Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI), and Bollinger Bands.
Preprocess the data and split it into training and testing sets.
Train and evaluate a Random Forest classifier and an LSTM model on the data.
Print the accuracies of both models, allowing you to compare their performance.
Generate buy/sell signals for each stock symbol using the Random Forest model. You can modify the generate_signals function to use the LSTM model if you find it performs better.
Note: The generated signals are for educational purposes only and should not be used for actual trading decisions without further research and validation.

Customization
You can customize the stock symbols, date range, and technical indicator parameters by modifying the corresponding variables in the stock_analysis.py script.

For example, to analyze a different set of stock symbols, change the symbols variable:
symbols = ['FB', 'NFLX', 'NVDA', 'ADBE', 'CRM']

To change the date range for stock data, modify the start_date and end_date variables:
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # Fetch data for the past 2 years

To adjust the parameters for the technical indicators, update the values in the corresponding function calls:
data['SMA'] = calculate_SMA(data['Close'], 21)  # Change SMA period to 21 days
data['EMA'] = calculate_EMA(data['Close'], 21)  # Change EMA period to 21 days
data['RSI'] = calculate_RSI(data['Close'], 14)  # Keep RSI period at 14 days
data['UpperBB'], data['LowerBB'] = calculate_BollingerBands(data['Close'], 30)  # Change Bollinger Bands period to 30 days
You can also adjust the parameters of the machine learning models and the training process to optimize their performance.
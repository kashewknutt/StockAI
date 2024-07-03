# train_model.py

import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
from .credentials import alpha_vantage_key

# Define constants
ALPHA_VANTAGE_API_KEY = alpha_vantage_key

def fetch_historical_data(symbol):
    """Fetch historical stock data from Alpha Vantage."""
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'outputsize': 'full'
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'Time Series (Daily)' in data:
        df = pd.DataFrame(data['Time Series (Daily)']).transpose()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        return df
    else:
        raise ValueError(f"Could not fetch data for symbol: {symbol}")

def preprocess_data(df):
    """Preprocess the data for training."""
    df['prev_close'] = df['close'].shift(1)
    df.dropna(inplace=True)
    X = df[['prev_close', 'open', 'high', 'low', 'volume']]  # Features
    y = df['close']  # Target variable
    return X, y

def train_model(X, y):
    """Train a linear regression model on the given data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained with Mean Squared Error: {mse}")
    return model

def save_model(model, symbol, save_dir='models'):
    """Save the trained model to a file."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{symbol}_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model for {symbol} saved to {model_path}")

def train_for_symbol(symbol):
    """Fetch, preprocess, train and save the model for the given stock symbol."""
    # Fetch historical data
    print(f"Fetching historical data for {symbol}...")
    df = fetch_historical_data(symbol)
    
    # Preprocess the data
    print("Preprocessing the data...")
    X, y = preprocess_data(df)
    
    # Train the model
    print("Training the model...")
    model = train_model(X, y)
    
    # Save the model
    print(f"Saving the model for {symbol}...")
    save_model(model, symbol)
    print(f"Model training and saving complete for {symbol}.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        stock_symbol = sys.argv[1].upper()  # Get the stock symbol from command line arguments
        train_for_symbol(stock_symbol)
    else:
        print("Please provide a stock symbol as a command line argument.")

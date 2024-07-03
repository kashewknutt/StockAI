# views.py
import joblib
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import os
from django.shortcuts import render
from django.http import JsonResponse
import requests
from bs4 import BeautifulSoup
import spacy

from .train_model import train_for_symbol
from stockAI.settings import BASE_DIR
from .companyDatabase import company_to_symbol
import difflib  # For finding similar company names
from .credentials import alpha_vantage_key

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

def lookup_stock_symbol(company_name):
    """
    Lookup stock symbol using Alpha Vantage API or another external service.
    """
    api_key = alpha_vantage_key  # Replace with your Alpha Vantage API key
    base_url = 'https://www.alphavantage.co/query'
    
    params = {
        'function': 'SYMBOL_SEARCH',
        'keywords': company_name,
        'apikey': api_key
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    print("Alpha response: ", data)
    if 'bestMatches' in data and data['bestMatches']:
        return data['bestMatches']
    
    return []

def predict_stock(symbol):
    """
    Predict the stock performance using the trained model.
    """
    model_path = os.path.join('models', f'{symbol}_model.pkl')
    
    if not os.path.exists(model_path):
        return f"No model found for {symbol}. Please train the model first."
    
    try:
        # Load the trained model
        model = joblib.load(model_path)
        
        # Fetch the latest stock data (for simplicity, using the last available data)
        df = fetch_historical_data(symbol)  # Reuse function from train_model.py
        latest_data = df.iloc[-1]  # Get the last row of data
        
        # Prepare the input data for prediction
        prev_close = latest_data['close']
        features = np.array([[prev_close, latest_data['open'], latest_data['high'], latest_data['low'], latest_data['volume']]])
        
        # Make the prediction
        predicted_close = model.predict(features)[0]
        
        # Assess the stock based on predicted change
        current_close = latest_data['close']
        change = predicted_close - current_close
        recommendation = "Buy" if change > 0 else "Hold/Sell"
        
        return f"Predicted closing price for {symbol}: {predicted_close:.2f}. Recommendation: {recommendation}."
    
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

# Reuse or import the fetch_historical_data function from train_model.py
def fetch_historical_data(symbol):
    """Fetch historical stock data from Alpha Vantage."""
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': alpha_vantage_key,
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

def extract_stock_symbol(query):
    """
    Extract the stock symbol or company name from the user's query.
    """
    doc = nlp(query.lower())  # Convert the query to lowercase for better consistency
    print("Parsed entities:", [(entity.text, entity.label_) for entity in doc.ents])

    # Financial keywords and common company query keywords
    financial_keywords = ["stock", "shares", "price", "market", "trade", "value", "company", "tell", "about"]
    
    # Preprocess query to handle common synonyms or variations
    normalized_query = query.upper().strip()
    
    # List of common company-related keywords and potential synonyms to improve matching
    common_company_keywords = ["INC", "CORP", "LTD", "COMPANY", "CORPORATION", "ENTERPRISES", "HOLDINGS", "GROUP"]
    
    # Step 1: Check if the user directly provides a stock symbol
    if "STOCK SYMBOL" in query.upper():
        # Extract the provided symbol
        possible_symbols = [word for word in normalized_query.split() if word.isupper() and word not in ["THE", "EXACT", "STOCK", "SYMBOL", "IS"]]
        if possible_symbols:
            return possible_symbols[0]
        
    if "EXACT SYMBOL" in query.upper():
        # Extract the provided symbol
        possible_symbols = [word for word in normalized_query.split() if word.isupper() and word not in ["IS", "THE", "EXACT", "SYMBOL"]]
        if possible_symbols:
            return possible_symbols[0]  # Return the symbol directly provided by the user
        
    if "SYMBOL" in query.upper():
        # Extract the provided symbol
        possible_symbols = [word for word in normalized_query.split() if word.isupper() and word not in ["IS", "THE", "EXACT", "SYMBOL"]]
        if possible_symbols:
            return possible_symbols[0]
    
    # Step 2: Use spaCy to extract entities
    for entity in doc.ents:
        if entity.label_ in ["ORG", "PRODUCT"]:
            potential_symbol = company_to_symbol.get(entity.text.upper().strip())
            if potential_symbol:
                return potential_symbol  # Return the mapped stock symbol
            else:
                return entity.text.upper().strip()  # Return the detected entity text
    
    # Step 3: Check for financial keywords in the context
    for token in doc:
        if token.text.lower() in financial_keywords:
            prev_token = token.nbor(-1) if token.i > 0 else None
            next_token = token.nbor(1) if token.i < len(doc) - 1 else None

            if prev_token and prev_token.ent_type_ in ["ORG", "PRODUCT"]:
                return prev_token.text.upper().strip()
            if next_token and next_token.ent_type_ in ["ORG", "PRODUCT"]:
                return next_token.text.upper().strip()
    
    # Step 4: Use common company names or symbols if no entities matched
    words = normalized_query.split()
    for word in words:
        if word in company_to_symbol:
            return company_to_symbol[word]

    # Step 5: Try to match using a more comprehensive approach
    for word in words:
        if word in company_to_symbol:
            return company_to_symbol[word]
        if any(keyword in word for keyword in common_company_keywords):
            possible_match = company_to_symbol.get(word)
            if possible_match:
                return possible_match
    
    # Step 6: If no match found, attempt to find the most similar known companies
    similar_companies = find_similar_companies(normalized_query)
    if similar_companies:
        return similar_companies  # Return the list of similar companies for user confirmation
    
    # Step 7: Use external API to look up the stock symbol dynamically
    matches = lookup_stock_symbol(normalized_query)
    if matches:
        return matches[0]['1. symbol']  # Return the best match symbol
    
    # Fallback: Ask the user to provide the exact company symbol
    return handle_ambiguous_query(normalized_query)

def find_similar_companies(company_name):
    """
    Find similar companies based on the input company name.
    """
    company_list = company_to_symbol.keys()
    similar_companies = difflib.get_close_matches(company_name.upper(), company_list, n=5, cutoff=0.5)
    return similar_companies

def handle_ambiguous_query(user_query):
    """
    Handle ambiguous queries by suggesting similar company names.
    """
    similar_companies = find_similar_companies(user_query)
    if similar_companies:
        return f"I'm not sure which company you mean. Did you mean one of these?\n" + "\n".join(similar_companies)
    else:
        return f"Sorry, I couldn't find any companies related to '{user_query}'. Please provide a more specific name or the exact stock symbol."

def chatbot(request):
    response = ""
    if request.method == 'POST':
        user_query = request.POST.get('query', '').strip()

        if user_query:
            # Extract the stock symbol from the query
            stock_symbol = extract_stock_symbol(user_query)  # Function to extract the stock symbol

            if stock_symbol:
                # Check if the model for the stock symbol already exists
                model_path = os.path.join('models', f'{stock_symbol}_model.pkl')
                if not os.path.exists(model_path):
                    # Train the model for the specified stock symbol
                    try:
                        print(f"Training model for {stock_symbol}...")
                        train_for_symbol(stock_symbol)  # Train and save the model
                        response = f"Model for {stock_symbol} has been trained and saved."
                    except Exception as e:
                        response = f"An error occurred while training the model for {stock_symbol}: {str(e)}"
                else:
                    response = f"Model for {stock_symbol} already exists."
                
                prediction = predict_stock(stock_symbol)
                response += f"\nPrediction for {stock_symbol}: {prediction}"

    return render(request, 'chatbot.html', {'response': response})

def scrape_stock_data(query):
    stock_symbol = query.upper().strip()
    url = f"https://finance.yahoo.com/quote/{stock_symbol}"
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    driver_path = os.path.join(BASE_DIR, 'chromedriver', 'chromedriver.exe')
    service = ChromeService(executable_path=driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)  # Increased wait time for better stability

        name_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[1]/div/section/h1')))
        price_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[2]/div[1]/section/div/section/div[1]/fin-streamer[1]/span')))
        volume_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/div[2]/ul/li[7]/span[2]/fin-streamer')))
        change_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[2]/div[1]/section/div/section/div[1]/fin-streamer[2]/span')))
        changepercent_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[2]/div[1]/section/div/section/div[1]/fin-streamer[3]/span')))
        marketcap_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/div[2]/ul/li[9]/span[2]/fin-streamer')))
        peratio_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/div[2]/ul/li[11]/span[2]/fin-streamer')))

        name = name_element.text.strip() if name_element else "N/A"
        price = price_element.text.strip() if price_element else "N/A"
        volume = volume_element.text.strip() if volume_element else "N/A"
        change = change_element.text.strip() if change_element else "N/A"
        changepercent = changepercent_element.text.strip() if changepercent_element else "N/A"
        marketcap = marketcap_element.text.strip() if marketcap_element else "N/A"
        peratio = peratio_element.text.strip() if peratio_element else "N/A"

        return {
            'name': name,
            'price': price,
            'volume': volume,
            'change': change,
            'changepercent': changepercent,
            'marketcap': marketcap,
            'peratio': peratio
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        driver.quit()

def format_stock_response(stock_data):
    return (
        f"Stock Name: {stock_data['name']}\n"
        f"Current Price: {stock_data['price']}\n"
        f"Volume: {stock_data['volume']}\n"
        f"Price Change: {stock_data['change']} ({stock_data['changepercent']})\n"
        f"Market Cap: {stock_data['marketcap']}\n"
        f"PE Ratio: {stock_data['peratio']}"
    )

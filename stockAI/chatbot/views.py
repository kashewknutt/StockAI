import os
import joblib
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
from bs4 import BeautifulSoup
from django.shortcuts import render
from django.http import JsonResponse
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
import difflib  # For finding similar company names
import json
import uuid  # Importing uuid module for generating unique user IDs

from .train_model import train_for_symbol, fetch_historical_data
from stockAI.settings import BASE_DIR
from .companyDatabase import company_to_symbol
from .credentials import alpha_vantage_key

# Initialize the transformers-based models
print("Initializing models...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
chatbot_pipeline = pipeline('question-answering', model="facebook/blenderbot-400M-distill")

nlp = spacy.load('en_core_web_sm')

def lookup_stock_symbol(company_name):
    """
    Lookup stock symbol using Alpha Vantage API or another external service.
    """
    print(f"Looking up stock symbol for: {company_name}")
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
    print(f"Predicting stock for symbol: {symbol}")
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

def extract_stock_symbol(query):
    """
    Extract the stock symbol or company name from the user's query.
    """
    print(f"Extracting stock symbol from query: {query}")
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
    
    # Step 2: Use transformers to extract entities
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model(**inputs).logits
    predictions = outputs.argmax(dim=2)
    predicted_entities = [
        (tokenizer.convert_ids_to_tokens(inputs.input_ids[0][i]), model.config.id2label[predictions[0][i].item()])
        for i in range(len(inputs.input_ids[0]))
    ]
    print("Predicted entities:", predicted_entities)

    for entity, label in predicted_entities:
        if label in ["ORG", "PRODUCT"]:
            potential_symbol = company_to_symbol.get(entity.upper().strip())
            if potential_symbol:
                return potential_symbol  # Return the mapped stock symbol
            else:
                return entity.upper().strip()  # Return the detected entity text
    
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
    print(f"Finding similar companies for: {company_name}")
    company_list = company_to_symbol.keys()
    similar_companies = difflib.get_close_matches(company_name.upper(), company_list, n=5, cutoff=0.5)
    return similar_companies

def handle_ambiguous_query(user_query):
    """
    Handle ambiguous queries by suggesting similar company names.
    """
    print(f"Handling ambiguous query: {user_query}")
    similar_companies = find_similar_companies(user_query)
    if similar_companies:
        return f"I'm not sure which company you mean. Did you mean one of these?\n" + "\n".join(similar_companies)
    else:
        return f"Sorry, I couldn't find any companies related to '{user_query}'. Please provide a more specific name or the exact stock symbol."

def scrape_stock_data(query):
    """
    Scrape stock data from Yahoo Finance for the given stock symbol.
    """
    print(f"Scraping stock data for symbol: {query}")
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

        name_element = wait.until(EC.presence_of_element_located((By.XPATH, "//h1[contains(@data-reactid,'7')]")))
        name = name_element.text.strip()
        price_element = wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(@data-reactid,'14')]")))
        price = price_element.text.strip()
        
        return f"{name} stock is currently priced at {price}."

    except Exception as e:
        return f"An error occurred while fetching stock data: {str(e)}"
    finally:
        driver.quit()

def chatbot(request):
    """
    Handle the chatbot interaction: both rendering the page and processing chat requests.
    """
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        print("Received user message:", user_message)
        # Create a unique user ID for session management
        user_id = request.session.session_key or str(uuid.uuid4())
        request.session['user_id'] = user_id  # Store the user ID in the session

        # Process the user's query and get the response
        response = process_query(user_message)
        print("Chatbot response:", response)

        return JsonResponse({'response': response})

    # For GET requests, render the chatbot page
    return render(request, 'chatbot.html')

def process_query(query):
    """
    Process the user's query to provide stock information or predictions.
    """
    print(f"Processing user query: {query}")
    if any(word in query.lower() for word in ['predict', 'forecast', 'estimate']):
        symbol = extract_stock_symbol(query)
        return predict_stock(symbol)

    if any(word in query.lower() for word in ['price', 'current', 'value', 'rate']):
        symbol = extract_stock_symbol(query)
        return scrape_stock_data(symbol)

    return handle_general_query(query)

def handle_general_query(query):
    """
    Handle general queries using the conversational pipeline.
    """
    context = "You are a stock trading helper. You need to understand what the user is talking about and then process and extract the requirement properly."
    print(f"Handling general query: {query}")
    print(f"context: ",{context})
    try:
        response = chatbot_pipeline(question=query, context=context)
        if response and len(response) > 0 and 'generated_text' in response[0]:
            generated_response = response[0]['generated_text']
            print("Generated chatbot response:", generated_response)
            return generated_response  # Return the generated response text
        else:
            return "Sorry, I couldn't generate a response to your query."

    except Exception as e:
        return f"Sorry, I encountered an error while processing your request: {str(e)}"

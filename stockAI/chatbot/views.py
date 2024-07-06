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
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, BertForQuestionAnswering, BertTokenizer
import difflib
import json
import uuid
from sentence_transformers import SentenceTransformer
import faiss
import cx_Oracle

from .train_model import train_for_symbol, fetch_historical_data
from stockAI.settings import BASE_DIR
from .companyDatabase import company_to_symbol
from .credentials import alpha_vantage_key, username, password, dsn

# Initialize the transformers-based models
print("Initializing models...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Load pre-trained Sentence Transformer model
print("Initializing Sentence Transformer model...")
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Replace Blenderbot with a more suitable model for Q&A and conversations
qa_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Initialize DialoGPT for conversational tasks
print("Initializing DialoGPT for conversation...")
conversation_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
conversation_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Connect to Oracle Database
connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
cursor = connection.cursor()

# Fetch stock embeddings from Oracle
def fetch_stock_embeddings():
    cursor.execute("SELECT stock_name, embedding FROM StockEmbeddings")
    results = cursor.fetchall()
    stock_names = [row[0] for row in results]
    embeddings = np.array([row[1] for row in results])
    return stock_names, embeddings

stock_names, embeddings = fetch_stock_embeddings()

# Build a FAISS index for the embeddings
dimension = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Other functions (lookup_stock_symbol, predict_stock, etc.) remain unchanged

def extract_stock_symbol_with_vector_search(query):
    """
    Extract the stock symbol or company name from the user's query using vector search.
    """
    print(f"Extracting stock symbol from query using vector search: {query}")
    query_embedding = sentence_model.encode([query])[0]  # Convert query to a vector
    distances, indices = index.search(np.array([query_embedding]), k=1)  # Find the nearest vector
    nearest_stock_name = stock_names[indices[0][0]]
    print(f"Identified stock name: {nearest_stock_name}")
    return nearest_stock_name

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
    model_path = os.path.join(BASE_DIR, 'models', f'{symbol}_model.pkl')
    
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
    print("Named entities identified by Spacy:", [(entity.text, entity.label_) for entity in doc.ents])

    financial_keywords = ["stock", "shares", "price", "market", "trade", "value", "company", "tell", "about"]

    normalized_query = query.upper().strip()
    print("Pre-Processed Query: ", normalized_query)

    common_company_keywords = ["INC", "CORP", "LTD", "COMPANY", "CORPORATION", "ENTERPRISES", "HOLDINGS", "GROUP"]

    if "STOCK SYMBOL" in query.upper() or "EXACT SYMBOL" in query.upper() or "SYMBOL" in query.upper():
        possible_symbols = [word for word in normalized_query.split() if word.isupper() and word not in ["THE", "EXACT", "STOCK", "SYMBOL", "IS"]]
        print("Possible symbols extracted directly from the query:", possible_symbols)
        if possible_symbols:
            return possible_symbols[0]

    try:
        print("Tokenizing the query for entity extraction...")
        inputs = tokenizer(query, return_tensors="pt")
        print("Tokenized input tensors:", inputs)

        if 'input_ids' not in inputs or inputs['input_ids'].size(0) == 0:
            print("No valid input tokens found. Handling as an ambiguous query.")
            return handle_ambiguous_query(normalized_query)

        print("Generating model outputs...")
        outputs = model(**inputs).logits
        print("Model output logits:", outputs)

        if not torch.is_tensor(outputs) or outputs.dim() != 3:
            print("Invalid or unexpected model output. Handling as an ambiguous query.")
            return handle_ambiguous_query(normalized_query)

        predictions = outputs.argmax(dim=2)
        print("Model predictions:", predictions)

        predicted_entities = [
            (tokenizer.convert_ids_to_tokens(inputs.input_ids[0][i]), model.config.id2label[predictions[0][i].item()])
            for i in range(len(inputs.input_ids[0]))
        ]
        print("Predicted entities from the model:", predicted_entities)

        for entity, label in predicted_entities:
            if label in ["ORG", "PRODUCT"]:
                potential_symbol = company_to_symbol.get(entity.upper().strip())
                print(f"Entity '{entity}' identified as {label}. Potential symbol: {potential_symbol}")
                if potential_symbol:
                    return potential_symbol
                else:
                    return entity.upper().strip()

        print("Checking for financial keywords in the context...")
        for token in doc:
            if token.text.lower() in financial_keywords:
                prev_token = token.nbor(-1) if token.i > 0 else None
                next_token = token.nbor(1) if token.i < len(doc) - 1 else None

                if prev_token and prev_token.ent_type_ in ["ORG", "PRODUCT"]:
                    print(f"Financial keyword '{token.text}' found with preceding entity '{prev_token.text}'")
                    return prev_token.text.upper().strip()
                if next_token and next_token.ent_type_ in ["ORG", "PRODUCT"]:
                    print(f"Financial keyword '{token.text}' found with following entity '{next_token.text}'")
                    return next_token.text.upper().strip()

        print("Matching common company names or symbols...")
        words = normalized_query.split()
        for word in words:
            if word in company_to_symbol:
                print(f"Common company or symbol matched: {word}")
                return company_to_symbol[word]

        print("Performing comprehensive matching for company names...")
        for word in words:
            if word in company_to_symbol:
                return company_to_symbol[word]
            if any(keyword in word for keyword in common_company_keywords):
                possible_match = company_to_symbol.get(word)
                if possible_match:
                    print(f"Matched possible company keyword: {word} -> {possible_match}")
                    return possible_match

        print("Using difflib for potential company name matching...")
        closest_match = difflib.get_close_matches(normalized_query, list(company_to_symbol.keys()), n=1, cutoff=0.6)
        if closest_match:
            print(f"Closest match found: {closest_match[0]}")
            return company_to_symbol[closest_match[0]]

        print("No direct match found. Handling as an ambiguous query.")
        return handle_ambiguous_query(normalized_query)

    except Exception as e:
        print(f"Error during entity extraction: {str(e)}. Handling as an ambiguous query.")
        return handle_ambiguous_query(normalized_query)

def handle_ambiguous_query(query):
    """
    Handle queries where no clear stock symbol or company name is identified.
    """
    print(f"Handling ambiguous query: {query}")
    possible_symbols = lookup_stock_symbol(query)
    if possible_symbols:
        return possible_symbols[0].get('1. symbol')
    return "Unknown"

def get_current_stock_price(symbol):
    """
    Scrape the current stock price from a financial website.
    """
    print(f"Fetching current stock price for: {symbol}")
    url = f"https://www.marketwatch.com/investing/stock/{symbol.lower()}"
    options = Options()
    options.headless = True
    service = ChromeService(executable_path='/path/to/chromedriver')  # Replace with your chromedriver path
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".intraday__price .value"))
        )
        price_element = driver.find_element(By.CSS_SELECTOR, ".intraday__price .value")
        price = price_element.text
        print(f"Retrieved stock price: {price}")
        return price
    except Exception as e:
        print(f"Error fetching stock price: {str(e)}")
        return "Error retrieving stock price."
    finally:
        driver.quit()
    
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
        user_message = json.loads(request.body).get('message', '').strip()  # Changed to load JSON from the body
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
    try:
        if any(word in query.lower() for word in ['predict', 'forecast', 'estimate']):
            symbol = extract_stock_symbol_with_vector_search(query)
            return predict_stock(symbol)

        if any(word in query.lower() for word in ['price', 'current', 'value', 'rate']):
            symbol = extract_stock_symbol_with_vector_search(query)
            return scrape_stock_data(symbol)

        return handle_general_query(query)

    except Exception as e:
        print("Error during query handling:", str(e))
        return f"Sorry, I encountered an error while processing your request: {str(e)}"

def handle_general_query(query, history=[]):
    """
    Handle general queries using the conversational pipeline.
    """
    print(f"Handling general query: {query}")
    context = "You are a stock trading helper. You need to understand what the user is talking about and then process and extract the requirement properly."

    try:
        # Tokenize the input along with conversation history
        input_ids = conversation_tokenizer.encode(query + conversation_tokenizer.eos_token, return_tensors='pt')

        # Append conversation history if available
        bot_input_ids = torch.cat([torch.tensor(history), input_ids], dim=-1) if history else input_ids
        
        # Generate a response from DialoGPT
        chat_history_ids = conversation_model.generate(bot_input_ids, max_length=1000, pad_token_id=conversation_tokenizer.eos_token_id)
        
        # Convert the output tokens to text
        response = conversation_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        print("Generated chatbot response:", response)
        
        # Update the conversation history
        history.append(chat_history_ids)
        
        return response

    except Exception as e:
        print("Error during query handling:", str(e))
        return f"Sorry, I encountered an error while processing your request: {str(e)}"
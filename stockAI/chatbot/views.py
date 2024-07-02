# views.py
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

from stockAI.settings import BASE_DIR
from .companyDatabase import company_to_symbol
import difflib  # For finding similar company names

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

def extract_stock_symbol(query):
    """
    Extract the stock symbol or company name from the user's query.
    """
    doc = nlp(query)
    print("Parsed entities:", [(entity.text, entity.label_) for entity in doc.ents])

    # Define more financial keywords and common company query keywords
    financial_keywords = ["stock", "shares", "price", "market", "trade", "value", "company", "tell", "about"]
    
    # Preprocess query to handle common synonyms or variations
    normalized_query = query.upper().strip()
    
    # List of common company-related keywords and potential synonyms to improve matching
    common_company_keywords = ["INC", "CORP", "LTD", "COMPANY", "CORPORATION", "ENTERPRISES", "HOLDINGS", "GROUP"]
    
    # Handle entity extraction
    for entity in doc.ents:
        if entity.label_ in ["ORG", "PRODUCT"]:
            potential_symbol = company_to_symbol.get(entity.text.upper().strip())
            if potential_symbol:
                return potential_symbol  # Return the mapped stock symbol
            else:
                return entity.text.upper().strip()  # Return the detected entity text
    
    # Handle keywords in the context of financial queries
    for token in doc:
        if token.text.lower() in financial_keywords:
            prev_token = token.nbor(-1) if token.i > 0 else None
            next_token = token.nbor(1) if token.i < len(doc) - 1 else None

            if prev_token and prev_token.ent_type_ in ["ORG", "PRODUCT"]:
                return prev_token.text.upper().strip()
            if next_token and next_token.ent_type_ in ["ORG", "PRODUCT"]:
                return next_token.text.upper().strip()
    
    # Step 3: Use common company names or symbols if no entities matched
    words = normalized_query.split()
    for word in words:
        if word in company_to_symbol:
            return company_to_symbol[word]

    # Try to match using a more comprehensive approach
    for word in words:
        if word in company_to_symbol:
            return company_to_symbol[word]
        if any(keyword in word for keyword in common_company_keywords):
            possible_match = company_to_symbol.get(word)
            if possible_match:
                return possible_match
    
    # If no match found, attempt to find the most similar known company
    similar_companies = find_similar_companies(normalized_query)
    if similar_companies:
        return similar_companies[0]  # Assume the closest match

    # Fallback: If no entities or predefined companies matched, return the query as-is
    print("Returning fallback:", normalized_query)
    return normalized_query

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
        return f"Sorry, I couldn't find any companies related to '{user_query}'. Please provide a more specific name."

def chatbot(request):
    response = ""
    if request.method == 'POST':
        user_query = request.POST.get('query', '').strip()

        if user_query:
            stock_symbol_or_name = extract_stock_symbol(user_query)

            if stock_symbol_or_name in company_to_symbol.values():
                stock_data = scrape_stock_data(stock_symbol_or_name)
                if stock_data:
                    response = format_stock_response(stock_data)
                else:
                    response = "Sorry, I couldn't find any data for your query."
            else:
                similar_companies = find_similar_companies(stock_symbol_or_name)
                if similar_companies:
                    response = f"I'm not sure which company you mean. Did you mean one of these?\n" + "\n".join(similar_companies)
                else:
                    response = f"Sorry, I couldn't find any companies related to '{user_query}'. Please provide a more specific name."

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

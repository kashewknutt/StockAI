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
from lxml import html

from stockAI.settings import BASE_DIR

import spacy
from .companyDatabase import company_to_symbol

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

def extract_stock_symbol(query):
    """
    Extract the stock symbol or company name from the user's query.
    """
    # Process the query with spaCy
    doc = nlp(query)

    # Print the parsed entities for debugging
    print("Parsed entities:", [(entity.text, entity.label_) for entity in doc.ents])

    # Define financial keywords that indicate a query about stock or financial information
    financial_keywords = ["stock", "shares", "price", "market", "trade", "value"]

    # Step 1: Extract company names or products first
    for entity in doc.ents:
        if entity.label_ in ["ORG", "PRODUCT"]:
            # Check if the entity is a known company or product
            potential_symbol = company_to_symbol.get(entity.text.upper().strip())
            if potential_symbol:
                return potential_symbol  # Return the mapped stock symbol
            else:
                return entity.text.upper().strip()  # Return the detected entity text

    # Step 2: Look for financial terms and their surrounding context
    for token in doc:
        if token.text.lower() in financial_keywords:
            # Check adjacent tokens for potential company names or symbols
            prev_token = token.nbor(-1) if token.i > 0 else None
            next_token = token.nbor(1) if token.i < len(doc) - 1 else None

            if prev_token and prev_token.ent_type_ in ["ORG", "PRODUCT"]:
                return prev_token.text.upper().strip()
            if next_token and next_token.ent_type_ in ["ORG", "PRODUCT"]:
                return next_token.text.upper().strip()

    # Step 3: Use a predefined list of common companies if no entities matched
    words = query.upper().strip().split()
    for word in words:
        if word in company_to_symbol:
            return company_to_symbol[word]

    # Step 4: If no entities or predefined companies matched, return the query as-is
    print("Returning fallback:", query.upper().strip())
    return query.upper().strip()




def chatbot(request):
    response = ""
    if request.method == 'POST':
        user_query = request.POST.get('query', '').strip()

        if user_query:
            stock_symbol_or_name = extract_stock_symbol(user_query)
            # Attempt to scrape stock data based on user query
            stock_data = scrape_stock_data(stock_symbol_or_name)
            if stock_data:
                response = format_stock_response(stock_data)
            else:
                response = "Sorry, I couldn't find any data for your query."

    return render(request, 'chatbot.html', {'response': response})

def scrape_stock_data(query):
    stock_symbol = query.upper().strip()
    url = f"https://finance.yahoo.com/quote/{stock_symbol}"
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (without GUI)
    #chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
    chrome_options.add_argument("--no-sandbox")  # Required for running as root
    #chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

    # Specify the path to the ChromeDriver
    driver_path = BASE_DIR / 'chromedriver' / 'chromedriver.exe'  # Adjust to your driver path
    service = ChromeService(executable_path=driver_path)

    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Open the URL
        driver.get(url)

        # Wait for the elements to be present
        wait = WebDriverWait(driver, 0)  # 2 seconds wait time

        name_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[1]/div/section/h1')))
        price_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[2]/div[1]/section/div/section/div[1]/fin-streamer[1]/span')))
        volume_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/div[2]/ul/li[7]/span[2]/fin-streamer')))
        change_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[2]/div[1]/section/div/section/div[1]/fin-streamer[2]/span')))
        changepercent_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[2]/div[1]/section/div/section/div[1]/fin-streamer[3]/span')))
        marketcap_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/div[2]/ul/li[9]/span[2]/fin-streamer')))
        peratio_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/div[2]/ul/li[11]/span[2]/fin-streamer')))

        # Extract the text content
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
        # Close the driver
        driver.quit()

def format_stock_response(stock_data):
    # Formats the scraped stock data into a readable response
    return (
        f"Stock Name: {stock_data['name']}\n"
        f"Current Price: {stock_data['price']}\n"
        f"Volume: {stock_data['volume']}\n"
        f"Price Change: {stock_data['change']} ({stock_data['changepercent']})\n"
        f"Market Cap: {stock_data['marketcap']}\n"
        f"PE Ratio: {stock_data['peratio']}"
    )
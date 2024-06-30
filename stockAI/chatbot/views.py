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

def chatbot(request):
    response = ""
    if request.method == 'POST':
        user_query = request.POST.get('query', '').strip()

        if user_query:
            # Attempt to scrape stock data based on user query
            stock_data = scrape_stock_data(user_query)
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
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
    chrome_options.add_argument("--no-sandbox")  # Required for running as root
    chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

    # Specify the path to the ChromeDriver
    driver_path = os.path.join(os.getcwd(), 'chromedriver')  # Adjust to your driver path
    service = ChromeService(executable_path=driver_path)

    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Open the URL
        driver.get(url)

        # Wait for the elements to be present
        wait = WebDriverWait(driver, 10)  # 10 seconds wait time

        name_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[1]/div/section/h1')))
        price_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[2]/div[1]/section/div/section[2]/div[1]/fin-streamer[1]/span')))
        volume_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/div[2]/ul/li[7]/span[2]/fin-streamer')))
        change_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[2]/div[1]/section/div/section[2]/div[1]/fin-streamer[2]/span')))
        changepercent_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[1]/div[2]/div[1]/section/div/section[2]/div[1]/fin-streamer[3]/span')))

        # Extract the text content
        name = name_element.text.strip()
        price = price_element.text.strip()
        volume = volume_element.text.strip()
        change = change_element.text.strip()
        changepercent = changepercent_element.text.strip()

        return {
            'name': name,
            'price': price,
            'volume': volume,
            'change': change,
            'changepercent': changepercent
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
        f"Price Change: {stock_data['change']} ({stock_data['changepercent']})"
    )
import spacy
import requests

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample mapping from company names to stock symbols
company_to_symbol = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "AMAZON": "AMZN",
    # Add more companies and their symbols here
}

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

# Testing the function
test_queries = [
    "What is the price of Apple shares?",
    "Tell me about Google stock",
    "Give me the latest on Amazon",
    "How is Microsoft doing today?",
    "What's the market value of Tesla?"
]

for query in test_queries:
    symbol = extract_stock_symbol(query)
    print(f"Query: {query} -> Stock Symbol/Company Name: {symbol}")

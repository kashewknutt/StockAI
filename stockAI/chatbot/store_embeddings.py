import os
import cx_Oracle
from sentence_transformers import SentenceTransformer
from .credentials import username, password, dsn

cx_Oracle.init_oracle_client(lib_dir=f"C:\\oracle\\instantclient_23_4", config_dir=f"C:\\oracle\\StockDB")

os.environ["PATH"] = "C:\\oracle\\instantclient_23_4\\instantclient_23_4" + ";" + os.environ["PATH"]
os.environ["TNS_ADMIN"] = "C:\\oracle\\StockDB"

# Step 2: Prepare and Store Data

# Part A: Collect Stock Data and Generate Embeddings
stock_data = [
    "Apple Inc.",
    "Microsoft Corporation",
    "Tesla Motors",
    "Amazon.com Inc.",
    "Facebook, Inc.",
    "Alphabet Inc.",
    "Netflix, Inc.",
    "NVIDIA Corporation",
    "Intel Corporation",
    "Advanced Micro Devices, Inc."
]

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to generate embeddings
def generate_embeddings(data):
    embeddings = model.encode(data)
    return embeddings

# Generate embeddings for the stock data
embeddings = generate_embeddings(stock_data)

print("Generated Embeddings:")
print(embeddings)

# Part B: Store Embeddings in Oracle Cloud

# Establish the database connection
connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
cursor = connection.cursor()

def prepare_embedding_for_storage(embedding):
    """
    Converts the embedding into a format suitable for storage in Oracle as SDO_GEOMETRY.
    """
    return embedding.tolist()  # Convert the numpy array to a list

# Prepare embeddings for each stock
prepared_embeddings = [prepare_embedding_for_storage(embedding) for embedding in embeddings]

for i, stock_name in enumerate(stock_data):
    embedding = prepared_embeddings[i]

    # Convert the list back to a comma-separated string
    embedding_str = ','.join(map(str, embedding))

    # Insert data into the database
    cursor.execute("""
        INSERT INTO StockEmbeddings (stock_name, embedding)
        VALUES (:1, SDO_GEOMETRY(:2))
    """, (stock_name, embedding_str))

# Commit the transaction to save changes
connection.commit()

print("Embeddings stored successfully in the Oracle database.")

# Verify the Data Storage (Optional)
cursor.execute("SELECT * FROM StockEmbeddings")
rows = cursor.fetchall()

for row in rows:
    print(f"Stock Name: {row[0]}, Embedding: {row[1]}")

# Close the cursor and connection
cursor.close()
connection.close()

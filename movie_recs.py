import pymongo
import os
import ssl
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try connecting to MongoDB with various SSL configurations
mongodb_uri = os.getenv("MONGODB_URI")

if not mongodb_uri:
    raise ValueError("MONGODB_URI not found in environment variables")

try:
    # First try with default settings
    client = pymongo.MongoClient(mongodb_uri, ssl_cert_reqs=ssl.CERT_NONE)
    # Test the connection
    client.admin.command('ping')
    print("MongoDB connection successful!")
except Exception as e:
    print(f"First connection attempt failed: {e}")
    print("Trying with alternative SSL configuration...")
    try:
        # Try with relaxed SSL settings
        client = pymongo.MongoClient(
            mongodb_uri,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )
        client.admin.command('ping')
        print("MongoDB connection successful with relaxed SSL!")
    except Exception as e2:
        print(f"All connection attempts failed: {e2}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Verify MongoDB Atlas IP whitelist allows your current IP")
        print("3. Check if MongoDB Atlas cluster is running")
        print("4. Try updating pymongo: pip install --upgrade pymongo")
        raise

db = client.sample_mflix
collection = db.movies


HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY not found in environment variables")

url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

def generate_embedding(text):
    response = requests.post(url, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")

    return response.json()

# import time

# count = 0
# for doc in collection.find({'plot': {'$exists': True}}).limit(50):
#     try:
#         print(f"Processing document {count + 1}/50: {doc.get('title', 'Unknown')}")
#         doc['plot_embedding_hf'] = generate_embeddings(doc['plot'])
#         collection.replace_one({'_id': doc['_id']}, doc)
#         count += 1
#         time.sleep(0.1)  # Small delay to avoid rate limiting
#     except Exception as e:
#         print(f"Error processing document {doc.get('_id')}: {e}")
#         continue

# print(f"Successfully processed {count} documents")

query = "imaginary characters from outer space at war"

results = collection.aggregate([ 
{
    "$vectorSearch": {
        "queryVector": generate_embedding(query),
        "path": "plot_embedding_hf",
        "numCandidates": 100,
        "limit": 4,
        "index": "default"
    }
}
]);

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')
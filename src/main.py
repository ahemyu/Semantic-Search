import os
import gradio as gr
import pandas as pd
import json
from data_preprocessing import preprocess_data
from embedding import run_embedding
from search import SemanticSearch

# choosable models for the embeddings
model_dimensions = {
    'all-MiniLM-L12-v2': 384,
    'all-MiniLM-L6-v2': 384,
    'msmarco-distilbert-base-v3': 768,
    'nli-mpnet-base-v2': 768
}

def load_config():
    """Load configuration settings from environment variables or fall back to config.json."""
    
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    
    # override model name with environment variable if set, and set vector dimension accordingly
    model_name = os.getenv('MODEL_NAME', config['model_name'])
    config['model_name'] = model_name
    config['vector_dimension'] = model_dimensions.get(model_name, config['vector_dimension'])
    
    return config

config = load_config()

def setup_environment():
    """Setup the environment by preprocessing data, generating embeddings, and initializing the search engine"""
    
    # preprocess the data and save the cleaned path
    cleaned_data_path = preprocess_data(config["data_file_path"])
    
    # run embedding generation based on the cleaned data
    run_embedding(cleaned_data_path, config["model_name"], config["vector_dimension"], config["collection_name"], f"http://{config['qdrant_host']}:{config['qdrant_port']}")
    
    # init the semantic search system with configured parameters
    searcher = SemanticSearch(config["model_name"], config["qdrant_host"], config["qdrant_port"], config["collection_name"])
    
    return searcher

# create a searcher instance by setting up the environment
searcher = setup_environment()

def search_products(query):
    """Defines the search functionality for Gradio interface"""
    
    # perform the search operation using the searcher instance
    results = searcher.search(query, limit=config["search_limit"])
    
    # Format the search results into a pandas DataFrame for display
    data = [{
        'Name': res['productName'],
        'Category': res['productCategory'],
        'Specifications': res['specifications']
    } for res in results]
    
    return pd.DataFrame(data)

# setup for the Gradio Interface
iface = gr.Interface(
    fn=search_products,
    inputs="text",
    outputs="dataframe",
    title="Product Search",
    description="Enter your query"
)
# launch the gradio app
iface.launch()

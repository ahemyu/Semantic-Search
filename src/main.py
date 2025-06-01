import os
import gradio as gr
import pandas as pd
import json
from typing import Dict, Any
from data_preprocessing import preprocess_data
from embedding import run_embedding
from search import SemanticSearch

# choosable models for the embeddings
model_dimensions = {
    'all-MiniLM-L12-v2': 384,
    'all-MiniLM-L6-v2': 384,
    'msmarco-distilbert-base-v3': 768,
    'nli-mpnet-base-v2': 768,
    "multi-qa-mpnet-base-cos-v1": 768,
    "all-mpnet-base-v2": 768,
}

def load_config() -> Dict[str, Any]:
    """Load configuration settings from environment variables or fall back to config.json."""
    
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    model_name = os.getenv('MODEL_NAME', config['model_name'])
    config['model_name'] = model_name
    config['vector_dimension'] = model_dimensions.get(model_name, config['vector_dimension'])
    
    return config

config = load_config()

def setup_environment() -> SemanticSearch:
    """Setup the environment by preprocessing data, generating embeddings, and initializing the search engine"""
    
    cleaned_data_path = preprocess_data(config["data_file_path"])
    run_embedding(cleaned_data_path, config["model_name"], config["vector_dimension"], config["collection_name"], f"http://{config['qdrant_host']}:{config['qdrant_port']}")
    searcher = SemanticSearch(config["model_name"], config["qdrant_host"], config["qdrant_port"], config["collection_name"])
    
    return searcher

searcher = setup_environment()

def search_products(query: str, limit: int = 5) -> pd.DataFrame:
    """Defines the search functionality for Gradio interface"""
    
    results = searcher.search(query, limit=limit)
    data = [{
        'Name': res['productName'],
        'Category': res['productCategory'],
        'Specifications': res['specifications']
    } for res in results]
    
    return pd.DataFrame(data)

iface = gr.Interface(
    fn=search_products,
    inputs=[
        gr.Textbox(label="Search Query", placeholder="Enter your query"),
        gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Results")
    ],
    outputs="dataframe",
    title=f"Using Model: {config['model_name']}",
    description="Enter your query",
    allow_flagging='never'
)

iface.launch()
